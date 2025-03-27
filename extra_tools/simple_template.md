# Results report : {{exp.name}}
## Hyperparameters:
{%for hyperparam in exp.hyperparameters%}
- **{{hyperparam[0]}}**: {{hyperparam[1]}}
{% endfor %}

## Weekly test results:
| Model | Week 1   |      | Week 2 |     | Week 3 |     | Week 4 |     | Week 5 |     | Week 6 |     |
|-------|----------|------|--------|-----|--------|-----|--------|-----|--------|-----|--------|-----|
|       |**MAE**| **F<sub>1</sub>**|**MAE**|**F<sub>1</sub>**| **MAE**|**F<sub>1</sub>**|**MAE**|**F<sub>1</sub>**| **MAE**|**F<sub>1</sub>**|**MAE**|**F<sub>1</sub>**|
| {{exp.name}} | {{exp.w_test_mae[0]}} | {{exp.w_test_f1[0]}} | {{exp.w_test_mae[1]}} | {{exp.w_test_f1[1]}} | {{exp.w_test_mae[2]}} | {{exp.w_test_f1[2]}} | {{exp.w_test_mae[3]}} | {{exp.w_test_f1[3]}} | {{exp.w_test_mae[4]}} | {{exp.w_test_f1[4]}} | {{exp.w_test_mae[5]}} | {{exp.w_test_f1[5]}} |

## Results over the test set:
| Model        | **MAE** | **RMSE** | **F<sub>1</sub>** | **ROC-AUC** |
|------------|------------|------------|------------|------------|
| {{exp.name}} | {{exp.test_mae}} | {{exp.test_rmse}} | {{exp.test_f1}} | {{exp.test_roc}} |

## Learning curves:
Learning curves can be analyzed by pointing a tensorboard session to the path : {{exp.tensorboard_path}}