from comet import download_model, load_from_checkpoint


if __name__ == "__main__":
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    print(model)
