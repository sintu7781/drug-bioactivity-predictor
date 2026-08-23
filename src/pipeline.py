from __future__ import annotations

import subprocess
import sys


def run_module(
    module: str
) -> None:
    
    print(
        f"\nRunning {module}..."
    )
    
    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
        ],
        check=True,
    )
    

def main() -> None:
    
    run_module(
        "src.download_data"
    )
    
    run_module(
        "src.download_molecules"
    )
    
    run_module(
        "src.curate"
    )
    
    run_module(
        "src.build_features"
    )
    
    run_module(
        "src.train"
    )
    
    run_module(
        "src.evaluate"
    )
    
    print(
        "\n================================"
    )
    
    print(
        "\nPipeline completed successfully."
    )
    
    print(
        "\n================================"
    )
    

if __name__ == "__main__":
    main()