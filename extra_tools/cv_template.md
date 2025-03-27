# Results report : 5 Fold CV Summary and t-paired test

## Summary results by fold:
| Model       | Baseline_LSTM |     |                  | HM | | |
|-------------|---------|----------|-------------------|----|-|-|
| **Fold**        | **MAE**    | **RMSE** | **F<sub>1</sub>**           | **MAE**         | **RMSE**      | **F<sub>1</sub>** |
|1                | {{exp.mae_lstm[0]}} | {{exp.rmse_lstm[0]}} | {{exp.f1_lstm[0]}} | {{exp.mae_hm[0]}}  | {{exp.rmse_hm[0]}} | {{exp.f1_hm[0]}} |
|2                | {{exp.mae_lstm[1]}} | {{exp.rmse_lstm[1]}} | {{exp.f1_lstm[1]}} | {{exp.mae_hm[1]}}  | {{exp.rmse_hm[1]}} | {{exp.f1_hm[1]}} |
|3                | {{exp.mae_lstm[2]}} | {{exp.rmse_lstm[2]}} | {{exp.f1_lstm[2]}} | {{exp.mae_hm[2]}}  | {{exp.rmse_hm[2]}} | {{exp.f1_hm[2]}} |
|4                | {{exp.mae_lstm[3]}} | {{exp.rmse_lstm[3]}} | {{exp.f1_lstm[3]}} | {{exp.mae_hm[3]}}  | {{exp.rmse_hm[3]}} | {{exp.f1_hm[3]}} |
|5                | {{exp.mae_lstm[4]}} | {{exp.rmse_lstm[4]}} | {{exp.f1_lstm[4]}} | {{exp.mae_hm[4]}}  | {{exp.rmse_hm[4]}} | {{exp.f1_hm[4]}} |

## Results of the t-paired test:
| Metric            | **t-statistic**        | **p-value**        | 
|-------------------|------------------------|--------------------|
| **MAE**           | {{exp.t_statistic[0]}} | {{exp.p_value[0]}} |
| **RMSE**          | {{exp.t_statistic[1]}} | {{exp.p_value[1]}} |
| **F<sub>1</sub>** | {{exp.t_statistic[2]}} | {{exp.p_value[2]}} |