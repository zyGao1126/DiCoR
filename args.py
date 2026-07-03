import argparse
import os


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="risbench", choices=["risbench", "rrsisd", "refsegrs"])
    parser.add_argument("--refer-data-root", "--refer_data_root", dest="refer_data_root", required=True)
    parser.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=480)
    parser.add_argument("--workers", "-j", type=int, default=8)
    parser.add_argument("--pin-mem", "--pin_mem", dest="pin_mem", action="store_true")
    parser.add_argument("--print-freq", type=int, default=20)

    parser.add_argument("--swin-type", "--swin_type", dest="swin_type", default="base", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--num-tmem", "--num_tmem", dest="num_tmem", type=int, default=3)
    parser.add_argument("--num-heads-fusion", "--num_heads_fusion", dest="num_heads_fusion", type=int, default=1)
    parser.add_argument("--window12", action="store_true")
    parser.add_argument("--ck-bert", "--ck_bert", dest="ck_bert", default="bert-base-uncased")
    parser.add_argument("--bert-tokenizer", "--bert_tokenizer", dest="bert_tokenizer", default="./bert-base-uncased")
    parser.add_argument(
        "--pretrained-swin-weights",
        "--pretrained_swin_weights",
        dest="pretrained_swin_weights",
        default="./pretrained_weights/swin_base_patch4_window12_384_22k.pth",
    )

    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--distributed", action="store_true")


def baseline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train the coarse DiCoR baseline")
    add_common_args(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", "--wd", dest="weight_decay", type=float, default=1e-2)
    parser.add_argument("--snapshot-epochs", default="10,15,20,30,39")
    return parser


def refiner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train the refiner from an offline prompt bank")
    add_common_args(parser)
    parser.add_argument("--coarse-ckpt", required=True)
    parser.add_argument("--prompt-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", "--wd", dest="weight_decay", type=float, default=1e-4)
    return parser


def localization_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Jointly train the localization guide with the coarse branch")
    add_common_args(parser)
    parser.add_argument("--coarse-ckpt", required=True)
    parser.add_argument("--prompt-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--guide-pretrain-epochs", type=int, default=40)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--guide-lr", type=float, default=1e-4)
    parser.add_argument("--evidence-lr", type=float, default=8e-4)
    parser.add_argument("--evidence-weight-decay", type=float, default=1e-4)
    parser.add_argument("--winner-lr", type=float, default=5e-4)
    parser.add_argument("--winner-weight-decay", type=float, default=1e-4)
    parser.add_argument("--weight-decay", "--wd", dest="weight_decay", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser


def test_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate DiCoR")
    add_common_args(parser)
    parser.add_argument("--coarse-ckpt", required=True)
    parser.add_argument("--refiner-ckpt", default="")
    parser.add_argument("--locate-ckpt", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--visual-dir", default="experiments/test")
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser
