"""ACT controller with the legacy pin or an explicitly validated model manifest."""
from scripts.deployment.run_contact_controller import main as _shared_main


def main(argv=None):
    return _shared_main(argv, policy_kind="act")


if __name__ == "__main__":
    raise SystemExit(main())
