import requests, json
import numpy as np

data = {
    "search_param" : {
        "continuous" : {
            "type" : "r",
            "range" : [-5, 5],
            "N" : 2
        },

        "ordinal" : {
            "type" : "i",
            "range" : [-100, 100],
            "N" : 1
        },

        "nominal" : {
            "type" : "c",
            "range" : ['OK', 'A', 'B', 'C', 'D', 'E'],
            "N" : 1
        }
    },

    "bo_param" : {
        "n_job" : 3,
        "n_point" : 3,
        "max_iter" : 20,
        "n_init_sample" : 3,
        "verbose" : True,
        "minimize" : True,
        "n_obj": 1
    }
}

address = 'http://207.246.97.250:7200'

def obj_func_dict_eval(par):
    """Example mix-integer test function
    with dictionary as input 
    """
    x_r = np.asarray([par[k] for k in par.keys() if k.startswith('continuous')])
    x_i, x_d = par['ordinal'], par['nominal']
    if x_d == 'OK':
        tmp = 0
    else:
        tmp = 1
    return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.

r = requests.post(address, json=data)
job_id = r.json()['job_id']

for i in range(10):  
    r = requests.get(address, params={'ask' : 'null', 'job_id' : job_id})
    tell_data = r.json()

    y = [obj_func_dict_eval(_) for _ in tell_data['X']]
    tell_data['y'] = y 
    
    r = requests.post(address, json=tell_data)
