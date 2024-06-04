import os
import cv2
import torch
import trimesh
import numpy as np
import progressbar
from PIL import Image
import torch.nn.functional as F
from pyhocon import ConfigFactory
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from models.gens import GenS
from models.losses.loss import Loss

from utils.distribute import *
from utils.tools import *
from utils.scheduler import WarmupCosineLR
from datasets import get_loader
from utils.clean_mesh import clean_mesh


class Runner:
    def __init__(self, args):
        cudnn.benchmark = True
        # os.environ['CUDA_LAUNCH_BLOCKING']='1'
        init_distributed_mode(args)

        self.distributed = args.distributed
        self.mode = args.mode
        self.device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

        self.conf = ConfigFactory.parse_file(args.conf)

        self.epochs = self.conf.get_int("train.epochs")
        self.base_exp_dir = self.conf["general.base_exp_dir"]
        if self.mode == "finetune":
            scene = self.conf["finetune_dataset.scene"] if args.scene is None else args.scene
            ref_view = self.conf["finetune_dataset.ref_view"] if args.ref_view is None else args.ref_view
            self.conf["finetune_dataset"]["ref_view"] = ref_view
            self.conf["finetune_dataset"]["scene"] = scene
            self.base_exp_dir = os.path.join(self.base_exp_dir, scene, f"view{ref_view}")
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.lr_confs = self.conf["train.lr_confs"]
        self.log_freq = self.conf.get_float("train.log_freq")
        self.save_freq = self.conf.get_float("train.save_freq")
        self.val_freq = self.conf.get_float("train.val_freq")
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.warmup = self.conf.get_float("train.warmup")
        self.alpha = self.conf.get_float("train.alpha")
        self.mesh_resolution = args.mesh_resolution
        self.clean_mesh = args.clean_mesh
        
        if is_main_process():
            log_dir = os.path.join(self.base_exp_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir, comment="Record network info")
            if self.mode == "train" or self.mode == "finetune":
                self.codes_backup()

        if self.mode == "finetune":
            self.finetune_dataset = get_loader(self.conf["finetune_dataset"], self.mode, False)
        else:
            if self.mode == "train":    
                self.train_loader, self.train_sampler, self.train_dataset = get_loader(self.conf["train_dataset"], self.mode, self.distributed)
            self.val_loader, self.val_sampler, self.val_dataset = get_loader(self.conf["val_dataset"], "val", self.distributed)
            
        self.model = GenS(self.conf["model"]).to(self.device)
        
        self.start_epoch = 0
        
        if args.resume is not None:
            # if you resume the finetuned model, you need specify the 'load_vol' commond
            if args.load_vol:
                self.model.load_params_vol(args.resume, self.device)
            else:
                print("Loading model...")
                ckpt = torch.load(args.resume, map_location="cpu")
                self.model.load_state_dict(ckpt["model"], strict=False)
                if self.mode == "train":
                    print("Loading optimizer, scheduler...")
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                    self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                    self.start_epoch = ckpt["epoch"] + 1
                
        if self.mode == "finetune":
            assert args.resume is not None, "Youe need resume a ckpt"
            print("Init volume...")
            self.model.eval()
            init_inputs = self.finetune_dataset.get_all_images()
            init_inputs = tocuda(init_inputs)
            self.model.init_volumes(init_inputs)

        if self.mode != "val":
            optim_param = self.model.get_optim_params(lr_confs=self.lr_confs)
            self.optimizer = torch.optim.Adam(optim_param)
            self.lr_scheduler = WarmupCosineLR(self.optimizer, self.epochs, self.warmup, self.alpha)

        self.loss = Loss(self.conf["train.loss"]).to(self.device)

        self.model_without_ddp = self.model
        if self.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[args.local_rank])#, find_unused_parameters=True)
            self.model_without_ddp = self.model.module

    def run(self):
        if self.mode == "train":
            self.train()
        elif self.mode == "val":
            self.validate()
        elif self.mode == "finetune":
            self.finetune()
        else:
            raise NotImplementedError("Not implemented mode {}!".format(self.mode))

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            self.train_epoch(epoch)

            if is_main_process() and (((epoch + 1) % self.save_freq == 0) or (epoch + 1) >= self.epochs):
                ckpt_dir = os.path.join(self.base_exp_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model': self.model_without_ddp.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict()},
                    "{}/model_{:0>3}.ckpt".format(ckpt_dir, epoch))
            
            if (epoch + 1) % self.val_freq == 0:
                self.validate(epoch)
            
            torch.cuda.empty_cache()
    
    def train_epoch(self, epoch):
        self.model.train()
        
        if is_main_process():
            pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                progressbar.Timer(), ",", progressbar.ETA(), ",", progressbar.Variable('LR', width=1), ",",
                progressbar.Variable('cl', width=1), ",", progressbar.Variable('ml', width=1), ",", 
                progressbar.Variable('dl', width=1), ",", progressbar.Variable('sl', width=1), ",", 
                progressbar.Variable('psnr', width=1)]
            pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.train_loader),
                                        prefix="Epoch {}/{}:".format(epoch, self.epochs)).start()
        
        avg_scalars = DictAverageMeter()
        
        for batch, inputs in enumerate(self.train_loader):
            inputs = tocuda(inputs)

            anneal_ratio = self.get_cos_anneal_ratio(epoch + batch / len(self.train_loader))
            outputs = self.model(self.mode, inputs, cos_anneal_ratio=anneal_ratio, step=epoch+batch/len(self.train_loader))
            
            psnr = 20.0 * torch.log10(1.0 / (((outputs["color_fine"] - inputs["color"])**2).mean()).sqrt())

            loss_res = self.loss(outputs, inputs, epoch + batch / len(self.train_loader))
            loss = loss_res["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step(epoch + batch / len(self.train_loader))

            scalar_opts = loss_res
            scalar_opts["psnr"] = psnr

            if self.distributed:
                scalar_opts = reduce_scalar_outputs(scalar_opts)
            
            scalar_opts = tensor2float(scalar_opts)
            avg_scalars.update(scalar_opts)

            if is_main_process():
                pbar.update(batch, LR=self.optimizer.param_groups[0]['lr'], 
                            cl="{:.3f}|{:.3f}".format(scalar_opts["color_loss"], avg_scalars.avg_data["color_loss"]),
                            ml="{:.3f}|{:.3f}".format(scalar_opts["mfc_loss"], avg_scalars.avg_data["mfc_loss"]),
                            dl="{:.3f}|{:.3f}".format(scalar_opts["depth_loss"], avg_scalars.avg_data["depth_loss"]),
                            sl="{:.3f}|{:.3f}".format(scalar_opts["pseudo_sdf_loss"], avg_scalars.avg_data["pseudo_sdf_loss"]),
                            psnr="{:.3f}|{:.3f}".format(scalar_opts["psnr"], avg_scalars.avg_data["psnr"])
                            )
                
                if batch >= len(self.train_loader) - 1:
                    save_scalars(self.writer, 'train_avg', avg_scalars.avg_data, epoch)

                if (batch + epoch * len(self.train_loader)) % int(self.log_freq * len(self.train_loader)) == 0:
                    save_scalars(self.writer, "train", scalar_opts, batch + epoch * len(self.train_loader))
                    
            del outputs
        
        if is_main_process():
            pbar.finish()
    
    @torch.no_grad()
    def validate(self, epoch=0):
        self.model.eval()
        
        if is_main_process():
            pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                progressbar.Timer(), ",", progressbar.ETA(), ",", progressbar.Variable('LR', width=1), ",",
                progressbar.Variable('cl', width=1), ",", progressbar.Variable('rdl', width=1), ",",
                progressbar.Variable('sdl', width=1), ",", progressbar.Variable('psnr', width=1)]
            pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.val_loader), prefix="Val:").start()

        avg_scalars = DictAverageMeter()

        for batch, inputs in enumerate(self.val_loader):
            inputs = tocuda(inputs)

            outputs = self.model("val", inputs, cos_anneal_ratio=1.0)

            file_name = inputs["file_name"]
            scale_mat = inputs["scale_mat"]
            scene = inputs["scene"]
            
            img_fine = outputs["img_fine"]
            normal_img = outputs["normal_img"]
            color_fine = outputs["color_fine"]
            sdf_depth = outputs["sdf_depth"]
            render_depth = outputs["render_depth"]
            vertices = outputs["vertices"]
            triangles = outputs["triangles"]
            
            mesh = trimesh.Trimesh(vertices, triangles)
            if self.clean_mesh:
                mesh = clean_mesh(mesh, inputs["masks"], inputs["intrs"], inputs["c2ws"])
            mesh.apply_transform(scale_mat.cpu().numpy())
            
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{}_epoch{}.ply'.format(scene, epoch)))
            
            os.makedirs(os.path.join(self.base_exp_dir, 'val_img'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'val_normal'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'val_sdf_depth'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'val_render_depth'), exist_ok=True)

            Image.fromarray(img_fine.astype(np.uint8)).save(os.path.join(self.base_exp_dir, 'val_img', '{}_epoch{}.png'.format(file_name, epoch)))
            Image.fromarray(normal_img.astype(np.uint8)).save(os.path.join(self.base_exp_dir, 'val_normal', '{}_epoch{}.png'.format(file_name, epoch)))
            self.save_depth(render_depth, os.path.join(self.base_exp_dir, 'val_render_depth', '{}_epoch{}.png'.format(file_name, epoch)))
            self.save_depth(sdf_depth, os.path.join(self.base_exp_dir, 'val_sdf_depth', '{}_epoch{}.png'.format(file_name, epoch)))
               
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - inputs["color"].cpu())**2).mean()).sqrt())

            color_loss = F.l1_loss(color_fine, inputs["color"].cpu())
            
            depth_ref = inputs["depth_ref"].cpu().numpy()
            skip = (depth_ref.shape[0] // render_depth.shape[0])
            depth_ref = depth_ref[::skip, ::skip]
            mask_ref = (depth_ref > 0).astype(np.float32)
            render_depth_loss = torch.tensor((np.abs(render_depth - depth_ref) * mask_ref).sum() / (mask_ref.sum() + 1e-8))
            sdf_depth_loss = torch.tensor((np.abs(sdf_depth - depth_ref) * mask_ref).sum() / (mask_ref.sum() + 1e-8))
            
            scalar_opts = {
                "color_loss": color_loss.to(self.device),
                "psnr": psnr.to(self.device),
                "render_depth_loss": render_depth_loss.to(self.device),
                "sdf_depth_loss": sdf_depth_loss.to(self.device)
            }

            if self.distributed:
                scalar_opts = reduce_scalar_outputs(scalar_opts)
            
            scalar_opts = tensor2float(scalar_opts)
            avg_scalars.update(scalar_opts)

            if is_main_process():
                pbar.update(batch,
                            cl="{:.3f}|{:.3f}".format(scalar_opts["color_loss"], avg_scalars.avg_data["color_loss"]),
                            rdl="{:.3f}|{:.3f}".format(scalar_opts["render_depth_loss"], avg_scalars.avg_data["render_depth_loss"]),
                            sdl="{:.3f}|{:.3f}".format(scalar_opts["sdf_depth_loss"], avg_scalars.avg_data["sdf_depth_loss"]),
                            psnr="{:.3f}|{:.3f}".format(scalar_opts["psnr"], avg_scalars.avg_data["psnr"]))
                
                if batch >= len(self.val_loader) - 1:
                    save_scalars(self.writer, 'val_img_avg', avg_scalars.avg_data, epoch)
        
        if is_main_process():
            pbar.finish()
            
    def finetune(self):
        self.model.train()
        
        pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                progressbar.Timer(), ",", progressbar.ETA(), ",", progressbar.Variable('LR', width=1), ",",
                progressbar.Variable('cl', width=1), ",", progressbar.Variable('ml', width=1), ",", 
                progressbar.Variable('psnr', width=1)]
        pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=self.epochs, prefix="Finetune:").start()
        
        avg_scalars = DictAverageMeter()
        image_perm = torch.randperm(self.finetune_dataset.num_views)
        for step in range(self.start_epoch, self.epochs):
            inputs = self.finetune_dataset.get_random_rays(image_perm[step % len(image_perm)])
            inputs = tocuda(inputs)
            
            anneal_ratio=self.get_cos_anneal_ratio(step)
            outputs = self.model("train", inputs, cos_anneal_ratio=anneal_ratio)
            
            loss_res = self.loss(outputs, inputs, step)
            loss = loss_res["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step(step)
            
            psnr = 20.0 * torch.log10(1.0 / (((outputs["color_fine"] - inputs["color"])**2).mean()).sqrt())
            
            scalar_opts = loss_res
            scalar_opts["psnr"] = psnr
            
            scalar_opts = tensor2float(scalar_opts)
            avg_scalars.update(scalar_opts)
            
            pbar.update(step, LR=self.optimizer.param_groups[0]['lr'], 
                cl="{:.3f}|{:.3f}".format(scalar_opts["color_loss"], avg_scalars.avg_data["color_loss"]),
                ml="{:.3f}|{:.3f}".format(scalar_opts["mfc_loss"], avg_scalars.avg_data["mfc_loss"]),
                psnr="{:.3f}|{:.3f}".format(scalar_opts["psnr"], avg_scalars.avg_data["psnr"])
                )
            
            if (step + 1) % self.log_freq == 0:
                save_scalars(self.writer, "finetune", scalar_opts, step)
                save_scalars(self.writer, 'finetune_avg', avg_scalars.avg_data, step)
                
            if (step + 1) % len(image_perm) == 0:
                image_perm = torch.randperm(self.finetune_dataset.num_views)
                
            if (((step + 1) % self.save_freq == 0) or (step + 1) >= self.epochs):
                finetune_params = self.model_without_ddp.get_params_vol()
                ckpt_dir = os.path.join(self.base_exp_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    'epoch': step,
                    'model': finetune_params,
                    'optimizer': self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict()},
                    "{}/model_{:0>3}.ckpt".format(ckpt_dir, step))
            
            if ((step + 1) % self.val_freq == 0) or ((step + 1) >= self.epochs):
                print("Val...")
                val_vid = 0
                val_inputs = self.finetune_dataset.get_rays_at(val_vid)
                val_inputs = tocuda(val_inputs)
                val_outputs = self.model("val", val_inputs, cos_anneal_ratio=1.0)
                scale_mat = val_inputs["scale_mat"]
                scene = val_inputs["scene"]
                
                img_fine = val_outputs["img_fine"]
                normal_img = val_outputs["normal_img"]
                sdf_depth = val_outputs["sdf_depth"]
                render_depth = val_outputs["render_depth"]
                vertices = val_outputs["vertices"]
                triangles = val_outputs["triangles"]
                
                mesh = trimesh.Trimesh(vertices, triangles)
                if self.clean_mesh:
                    mesh = clean_mesh(mesh, inputs["masks"], inputs["intrs"], inputs["c2ws"])
                mesh.apply_transform(scale_mat.cpu().numpy())
                
                os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
                mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{}_step{}.ply'.format(scene, step)))
                
                os.makedirs(os.path.join(self.base_exp_dir, 'val_img'), exist_ok=True)
                os.makedirs(os.path.join(self.base_exp_dir, 'val_normal'), exist_ok=True)
                os.makedirs(os.path.join(self.base_exp_dir, 'val_sdf_depth'), exist_ok=True)
                os.makedirs(os.path.join(self.base_exp_dir, 'val_render_depth'), exist_ok=True)

                Image.fromarray(img_fine.astype(np.uint8)).save(os.path.join(self.base_exp_dir, 'val_img', '{}_step{}.png'.format(val_vid, step)))
                Image.fromarray(normal_img.astype(np.uint8)).save(os.path.join(self.base_exp_dir, 'val_normal', '{}_step{}.png'.format(val_vid, step)))
                self.save_depth(render_depth, os.path.join(self.base_exp_dir, 'val_render_depth', '{}_step{}.png'.format(val_vid, step)))
                self.save_depth(sdf_depth, os.path.join(self.base_exp_dir, 'val_sdf_depth', '{}_step{}.png'.format(val_vid, step)))
                
        pbar.finish()
    
    def save_depth(self, depth, file_path):
        import matplotlib as mpl
        import matplotlib.cm as cm
        from PIL import Image
        
        # vmax = np.percentile(depth, 95)
        # vmin = depth.min()
        vmax = 2.5
        vmin = 0
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
        im = Image.fromarray(colormapped_im)
        im.save(file_path)
    
    def get_cos_anneal_ratio(self, step):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, step / self.anneal_end])
        
    def codes_backup(self):
        record_path = os.path.join(self.base_exp_dir, "codes_recording")
        os.makedirs(record_path, exist_ok=True)
        os.system("cp -r . "+record_path)