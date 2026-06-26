import os
import base64
import zipfile
import io


TARGET_DIR = "/home/youssef_mohammad/projects/mtc-aic4-phase2/lib"


def pack_models():
    buffer = io.BytesIO()

    # Zip the target directory, preserving paths relative to its parent so
    # that extraction yields lib/<...> at the destination.
    parent_dir = os.path.dirname(TARGET_DIR)
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(TARGET_DIR):
            for file in files:
                if file.endswith(".pyc") or "__pycache__" in root:
                    continue

                file_path = os.path.join(root, file)
                # Store paths relative to the parent of TARGET_DIR
                arcname = os.path.relpath(file_path, parent_dir)

                zf.write(file_path, arcname)

    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Save the payload to a text file
    with open("models_payload.txt", "w") as f:
        f.write(b64_str)

    print(f"SUCCESS: lib/ directory packed into models_payload.txt")
    print(f"  Target : {TARGET_DIR}")
    file_count = sum(
        1
        for r, _, fs in os.walk(TARGET_DIR)
        for fn in fs
        if not fn.endswith(".pyc") and "__pycache__" not in r
    )
    print(f"  Files  : {file_count}")
    print(f"  Size   : {len(b64_str):,} chars (base64)")


if __name__ == "__main__":
    pack_models()
