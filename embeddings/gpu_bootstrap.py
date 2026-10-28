import argparse
import os
from dataclasses import dataclass


DEFAULT_N_GPU = 1


@dataclass(frozen=True)
class GpuBootstrapResult:
    n_gpu: int
    use_safe_gpu: bool
    visible_devices: str | None
    configured: bool = True


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f'must be at least 1, got {value}')
    return number


def add_gpu_arguments(parser, multi_gpu=False):
    if multi_gpu:
        parser.add_argument(
            '--n-gpu',
            type=positive_int,
            default=DEFAULT_N_GPU,
            help='Number of GPUs to use (default: %(default)s).',
        )
    else:
        parser.add_argument(
            '--n-gpu',
            type=int,
            choices=(DEFAULT_N_GPU,),
            default=DEFAULT_N_GPU,
            help='Number of GPUs to use. Only single-GPU training is currently supported.',
        )
    parser.add_argument(
        '--use-safe-gpu',
        action='store_true',
        help='Reserve --n-gpu available GPUs with safe_gpu before CUDA is initialized.',
    )


def bootstrap_single_gpu(argv=None, environment=None, claim_gpus=None):
    """Configure single-GPU visibility before importing a CUDA framework."""
    return bootstrap_gpus(argv, environment=environment, claim_gpus=claim_gpus)


def bootstrap_gpus(argv=None, environment=None, claim_gpus=None, multi_gpu=False):
    """Configure visibility of --n-gpu GPUs before importing a CUDA framework.

    Without safe_gpu, the first --n-gpu devices of CUDA_VISIBLE_DEVICES are kept
    (devices 0..n-1 when it is unset). multi_gpu=False accepts only --n-gpu 1.
    """
    argv = list(argv) if argv is not None else None
    environment = os.environ if environment is None else environment

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_gpu_arguments(parser, multi_gpu=multi_gpu)
    args, _ = parser.parse_known_args(argv)

    # Displaying help must not reserve a GPU merely to print the full parser.
    if argv is not None and any(argument in ('-h', '--help') for argument in argv):
        return GpuBootstrapResult(
            n_gpu=args.n_gpu,
            use_safe_gpu=args.use_safe_gpu,
            visible_devices=environment.get('CUDA_VISIBLE_DEVICES'),
            configured=False,
        )

    if args.use_safe_gpu:
        if claim_gpus is None:
            try:
                from safe_gpu import safe_gpu
            except ImportError as exc:
                raise RuntimeError(
                    '--use-safe-gpu requires the optional safe-gpu package. '
                    'Install it or run without --use-safe-gpu.'
                ) from exc
            claim_gpus = safe_gpu.claim_gpus
        claim_gpus(args.n_gpu)
    else:
        visible_devices = environment.get('CUDA_VISIBLE_DEVICES')
        if visible_devices is None:
            environment['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(args.n_gpu))
        elif visible_devices:
            devices = [device.strip() for device in visible_devices.split(',')]
            if len(devices) < args.n_gpu:
                raise RuntimeError(f'--n-gpu {args.n_gpu} requested, but CUDA_VISIBLE_DEVICES='
                                   f'{visible_devices!r} lists only {len(devices)} device(s).')
            environment['CUDA_VISIBLE_DEVICES'] = ','.join(devices[:args.n_gpu])

    return GpuBootstrapResult(
        n_gpu=args.n_gpu,
        use_safe_gpu=args.use_safe_gpu,
        visible_devices=environment.get('CUDA_VISIBLE_DEVICES'),
    )
