def save_with_accelerate(accelerator, model, output_dir):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(
        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
    )