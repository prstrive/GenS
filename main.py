import argparse
from runner import Runner


parser = argparse.ArgumentParser(description="GenS args")

parser.add_argument('--conf', type=str, default='./confs/gens.conf')
parser.add_argument('--mode', type=str, default='train', choices=["train", "val", "finetune"])
parser.add_argument('--resume', type=str, help='checkpoint to resume')
parser.add_argument('--mesh_resolution', type=int, default=512)
parser.add_argument("--clean_mesh", action="store_true")
parser.add_argument('--scene', type=str, default=None, help='for finetuning')
parser.add_argument('--ref_view', type=int, default=None, help='for finetuning')
parser.add_argument("--load_vol", action="store_true", help="for loading finetuned model")

parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

if __name__ == "__main__":
    runner = Runner(args)
    runner.run()
