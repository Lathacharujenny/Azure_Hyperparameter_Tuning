from azureml.core import Workspace
from azureml.core import Environment, Experiment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, choice, PrimaryMetricGoal, MedianStoppingPolicy
from azureml.core import Model

ws = Workspace.from_config()


compute_target_name = 'Instance'
compute_target = ComputeTarget(name='Instance', workspace=ws)


env = Environment('environment')
packages = CondaDependencies.create(conda_packages=['scikit-learn','pip'],
                                    pip_packages=['azureml-defaults','azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages
script_config = ScriptRunConfig(source_directory='.',
                               script='train.py',
                               environment=env,
                               compute_target=compute_target)


params = GridParameterSampling(
    {

        '--learning_rate': choice(0.01, 0.1, 1.0),

        '--n_estimators' : choice(10, 100)
    }

)
median_terimination_policy = MedianStoppingPolicy(evaluation_interval=1)

hyperdrive = HyperDriveConfig(run_config=script_config, 

                          hyperparameter_sampling=params, 

                          policy=median_terimination_policy, # No early stopping policy

                          primary_metric_name='AUC', # Find the highest AUC metric

                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 

                          max_total_runs=6, # Restict the experiment to 6 iterations

                          max_concurrent_runs=2) # Run up to 2 iterations in parallel
experiment = Experiment(workspace=ws, name='mslearn-diabetes-hyperdrive')

run = experiment.submit(config=hyperdrive)


# Show the status in the notebook as the experiment runs
run.wait_for_completion(show_output=True)
for child_run in run.get_children_sorted_by_primary_metric():
    print(child_run)
    
best_run = run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
script_arguments = best_run.get_detailes()['runDefinition']['arguments']
print('Best Run Id: ', best_run.id)
print(' -AUC:', best_run_metrics['AUC'])
print(' -Accuracy:', best_run_metrics['Accuracy'])
print(' -Arguments:',script_arguments)



# Register model

best_run.register_model(model_path='outputs/diabetes_model.joblib', model_name='diabetes_model',

                        tags={'Training context':'Hyperdrive'},

                        properties={'AUC': best_run_metrics['AUC'], 'Accuracy': best_run_metrics['Accuracy']})