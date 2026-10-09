"""ACT checkpoint/GPU validation using synthetic observations; no devices."""
from scripts.deployment.check_contact_checkpoint import main as _shared_main


def main(argv=None):
    return _shared_main(argv, model_kind="act")


if __name__ == "__main__":
    main()
