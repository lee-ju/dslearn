import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings

def lm_stat(model, X, y, alternative="two_sided", col_names=None, digits=3):
  """
  @params \n
  model : sklearn.linearmodel.LinearRegression().fit(X, y) \n
  X : independent variables of model \n
  y : dependent variable of model \n
  alternative : a character string specifying the alternative hypothesis, must be one of "two_sided" (default), "greater" or "less". You can specify just the initial letter.
  """
  params = np.append(model.intercept_, model.coef_)
  pred = model.predict(X)
  N = X.shape[0]
  newX = pd.DataFrame({'Constant': np.ones(len(X))}).join(pd.DataFrame(X))
  MSE = (sum((y-pred)**2)) / (len(newX) - len(newX.columns))

  var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
  sd_b = np.sqrt(var_b)
  ts_b = params / sd_b

  if (alternative == 'two_sided') or (alternative == 't'):
    p_val = [2*(1-stats.t.cdf(np.abs(ts), N-2)) for ts in ts_b]
  elif (alternative == 'greater') or (alternative == 'g'):
    p_val = [(1-stats.t.cdf(ts, N-2)) for ts in ts_b]
  elif (alternative == 'less') or (alternative == 'l'):
    p_val = [(stats.t.cdf(ts, N-2)) for ts in ts_b]
  else:
    print("ERROR: \nChoose the SPECIFIC parameter for 'alternative'")

  sd_b = np.round(sd_b, digits)
  ts_b = np.round(ts_b ,digits)

  p_val = np.round(p_val, digits)
  params = np.round(params, digits)

  df = pd.DataFrame()
  df['coef'], df['se'], df['t_val'], df['p_val'] = [params, sd_b, ts_b, p_val]

  if col_names == None:
    col_names = ['Beta'+str(i+1) for i in range(X.shape[1])]

  col_names.insert(0, 'Intercept')
  df.index = col_names
  return df

def lm_r2(model, X, y, adjust=True, digits=3):
  N, p = X.shape
  pred = model.predict(X)
  ssr = sum((y-pred)**2)
  sst = sum((y-np.mean(y))**2)

  r2 = 1 - (float(ssr)) / sst

  if adjust:
    adj_r2 = 1 - (1-r2) * (N-1) / (N-p-1)
    return round(float(adj_r2), digits)
  else:
    return round(float(r2), digits)

def stepwise(X, y, model_type='linear', thred=0.05, variables=None):
  """
  @params \n
  X : independent variables \n
  y : dependent variable \n
  model_type : 'linear' (for Linear regression by default) or 'logit' (for Logistic regression)
  thred : p-value's threshold for stepwise selection. (default) 0.05
  variables : (list) column names of X
  """
  warnings.filterwarnings("ignore")
  if variables == None:
    X = pd.DataFrame(X)
    variables = ['V'+str(v) for v in range(X.shape[1])]
    X.columns = variables
  else:
    X = pd.DataFrame(X, columns=variables)

  selected = []
  
  #sv_per_step = []
  #adj_r2 = []
  #steps = []
  #step = 0

  while len(variables) > 0:
    remained = list(set(variables) - set(selected))
    pval = pd.Series(index=remained)
    for col in remained:
      x = X[selected + [col]]
      x = sm.add_constant(x)

      if model_type == 'linear':
        model = sm.OLS(y, x).fit(disp=0)
      elif model_type == 'logit':
        model = sm.Logit(y, x).fit()

      pval[col] = model.pvalues[col]

    min_pval = pval.min()
    if min_pval < thred:
      selected.append(pval.idxmin())
      
      while len(selected) > 0:
        selected_X = X[selected]
        selected_X = sm.add_constant(selected_X)

        if model_type == 'linear':
          selected_pval = sm.OLS(y, selected_X).fit(disp=0).pvalues[1:]
        elif model_type == 'logit':
          selected_pval = sm.Logit(y, selected_X).fit().pvalues[1:]

        max_pval = selected_pval.max()

        if max_pval >= thred:
          remove_variable = selected_pval.idxmax()
          selected.remove(remove_variable)
        else:
          break

      #step += 1
      #steps.append(step)
      #adj_r2_val = sm.OLS(y_train, sm.add_constant(X_train[selected])).fit(disp=0).rsquared_adj
      #adj_r2.append(adj_r2_val)
      #sv_per_step.append(selected.copy())
    else:
      break
  return selected

if __name__ == '__main__':
  from dslearn import multi_stat

  # Load Dataset
  from sklearn.datasets import load_diabetes
  X, y = load_diabetes(return_X_y=True)

  # Statistical Test (Beta1 = 0)
  from sklearn.linear_model import LinearRegression
  lm = LinearRegression().fit(X, y)
  print(multi_stat.lm_stat(model=lm, X=X, y=y))

  # get R2 or adj-R2
  print("R2 =", multi_stat.lm_r2(model=lm, X=X, y=y, adjust=False))
  print("adj-R2 =", multi_stat.lm_r2(model=lm, X=X, y=y, adjust=True))

  # Feature Selection (with Linear Regression)
  print("Selected variables:", stepwise(X=X, y=y))

  # Feature Selection (with Logistic Regression)
  print("Selected variables:", stepwise(X=X, y=y, model_type='logit'))