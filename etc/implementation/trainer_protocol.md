# Trainer Command Protocol

This file contains information regarding the command protocol of the Trainer class.

## Protocol Structure

- All commands are derived from a python dictionary that has been converted to json format
- The dictionary will contain 2 main keys
	- command_name
	- command_kwargs
- The trainer class will have functions that corresponds to the command_name and the command_kwargs will be the variables to pass into those functions

## Commands

### Select model

Select the model to be trained, 2 models can be selected simultaneously, the actor model and the critic model.

```
dict(
	command_name = select_model,
	command_kwargs = dict(
		model_type = <model.utils.ModelType.name>,
		model_name = <str(name)>,
		model_id = <str(datetime)>,
	)
)
```

### Create model

Create an entirely new model and select that model.

```
dict(
	command_name = create_model,
	command_kwargs = dict(
		model_type = <model.utils.ModelType.name>,
		model_name = <str(name)>,
		model_architecture = <model.model.MimeticSNAIL kwargs>
	)
)
```

### Set Optimizer

Set the optimizer to be used to train the model.

```
dict(
	command_name = set_optimizer,
	command_kwargs = dict(
		model_type = <model.utils.ModelType.name>,
		optimizer_dict = dict(
			optimizer_name = <name of optimizers in torch.optim>,
			optimizer_kwargs = <kwargs for optimizer>
		)
	)
)
```


### Start training

Start the training of the selected actor model.

- The model will be saved every <save_every>
- When the model reached <max_epochs>, the model will be automatically saved and paused.

```
dict(
	command_name = start_training,
	command_kwargs = dict(
		training_type =  <IL | RL>,
		save_every = <int>,
		max_epochs = <int>
		additional_algorithm_kwargs = dict(**kwargs)
	)
)
```

### Pause training

Pause the training of the actor model.

``` 
dict(
	command_name = pause_training,
	command_kwargs = None
)
```

### Stop training

Stop the training of the model(s). This command is different from pause in that the model will also be deselected.

```
dict(
	command_name = stop_training,
	command_kwargs = None
)
```

### Restart model

Restarts the weights of the model.

```
dict(
	command_name = restart_model,
	command_kwargs = dict(
		model_type = <model.utils.ModelType.name>
	)
)
```

### Save model

Saves the selected model.

```
dict(
	command_name = save_model,
	command_kwargs = dict(
		model_type = <model.utils.ModelType.name>
	)
)
```

### Select training data

Selects the task/demo episodes to be used for training.

```
dict(
	command_name = select_training_data,
	command_kwargs = dict(
		list_of_names = list(task/demo names)
	)
)
```

### Select device

Selects the device to train the model on.

```
dict(
	command_name = select_device,
	command_kwargs = dict(
		device_name = <cpu, gpu:0, etc.>
	)
)
```