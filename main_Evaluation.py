from __future__ import annotations

from app.controller.evaluation_controller import main as controller_main


def main() -> None:
    """Run evaluation on all processed images."""
    print("Starting reflection removal evaluation...")
    print("This will compare original images with processed results.")
    print()
    
    controller_main()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()