import pycaret
from pycaret.regression import *
from pycaret.datasets import get_data



dataset = get_data("insurance")

data = dataset.sample(frac=0.8, random_state=143)
data_unseen = dataset.drop(data.index)

s = setup(data = data, target='charges')

best = compare_models(sort='RMSE')

save_model(best,'insurance')

saved_final = load_model('insurance')

new_prediction = predict_model(saved_final, data=data_unseen)




