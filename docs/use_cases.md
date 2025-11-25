Use cases for probelab:

- Be able to define datasets and get tokenized versions of the data with good mask support.
  - Easy and ergonomic to add new datasets.
  - Easy to add new masks.
  - Easy to visualize masks.
- Training multiple pipelines on a model's layers. Pipelines compose preprocessing steps (layer selection, aggregation) with probes. All pipelines take the same input type (Activations object).
  - Be able to train multiple pipelines at the same time on different layers without having to recompute the activations.
  - Explicit 2-step workflow: (1) collect activations, (2) train pipelines on those activations.
  - Preprocessing is explicit in pipeline definition, enabling clear experimentation with different strategies.
  - Core workflow functions (`train_pipelines`, `evaluate_pipelines`) work with pre-collected activations.
  - Convenience functions (`train_from_model`, `evaluate_from_model`) handle both steps in one call.
- Using a pipeline to generate signal for model finetuning (i.e. a backpropable signal that can be used to finetune the model).
  - Pipelines should have a `.predict()` method that returns a tensor of predictions. The `.predict()` op should be differentiable.

It's important to focus on:

- **Simplicity and clarity**: The 2-step API (collect → train) makes it obvious what's happening. A researcher should understand that activations are collected once, then multiple pipelines can be trained on those activations.
- **Explicit over implicit**: Preprocessing steps are explicitly defined in the pipeline, not hidden in probe parameters.
- **Efficiency**: The library should be efficient and use as little memory as possible. The explicit 2-step workflow enables:
  - Collecting activations once for multiple pipelines
  - Clear control over what gets collected and when
  - Easy switching between in-memory and streaming modes
- **Composability**: Pipelines enable mixing and matching preprocessing transformers (SelectLayer, AggregateSequences, AggregateTokenScores) with different probe types.
