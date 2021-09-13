import json
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from crossindex_main import CrossIndex

experiment_config = {
    "flights":{
        "dimensions": ["ARR_DELAY","DEP_DELAY","DISTANCE","AIR_TIME","ARR_TIME","DEP_TIME"],
        "types": ["numerical","numerical","numerical","numerical","numerical","numerical"],
        "bin_width": [5,5,200,20,1,1],
        "offset": [-20,-20,0,0,0,0]
    },
    "movies":{
        "dimensions": ["Running_Time_min","US_Gross","IMDB_Rating","Production_Budget","US_DVD_Sales","Rotten_Tomatoes_Rating","Worldwide_Gross","Release_Date"],
        "types": ["numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical"],
        "bin_width": [20.0,5.0,0.5,2.0,2.0,5.0,5.0,30541006451.612904],
        "offset": [0,0,0,0,0,0,0,315550800000]
    },
    "weather":{
        "dimensions": ["TEMP_MIN","TEMP_MAX","SNOW","ELEVATION","LONGITUDE","PRECIPITATION","WIND","RECORD_DATE"],
        "types": ["numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical"],
        "bin_width": [5.0,5.0,50.0,200.0,20.0,0.5,0.5,163725000.0],
        "offset": [-10,-10,0,-200,-160,0,0,1325307600000]
    }
}

dataset_input_dir={
    "flights":{
        "1M":"data/Flights/dataset_flights_1M.csv",
        "10M":"data/Flights/dataset_flights_10M.csv",
        "100M":"data/Flights/dataset_flights_100M.csv"
    },
    "movies":{
        "1M":"data/Movies/dataset_movies_1M_fixed.csv",
        "10M":"data/Movies/dataset_movies_10M_fixed.csv",
        "100M":"data/Movies/movies_100M.csv"
    },
    "weather":{
        "1M":"data/Weather/dataset_weather_1M_fixed.csv",
        "10M":"data/Weather/dataset_weather_10M_fixed.csv",
        "100M":"data/Weather/weather_100M.csv"
    }
}


if __name__ == '__main__':
    res = {}
    datasets = ["flights", "movies", "weather"]
    for dataset in datasets:
        print('start dataset', dataset)
        tmp_res = {}
        config = experiment_config[dataset]
        sizes = ["1M","10M","100M"]
        for size in sizes:
            print('start building %s_%s'%(dataset,size))
            crossindex = CrossIndex(dataset+"_"+size, config["dimensions"], config["types"])
            crossindex.adjust_by_cardinality(dataset_input_dir[dataset][size], config["bin_width"], config["offset"], ",", False)
            time_cost = crossindex.build_csv()
            
            tmp = {}
            tmp["time_cost"] = time_cost
            tmp["dimensions"] = crossindex.dimensions
            tmp["bin_count"] = crossindex.bin_count
            tmp_res[size] = tmp
        res[dataset] = tmp_res
        print('%s finished.'%dataset)
    
    print('Offline cost experiment ends.')

    with open("offline_cost.json", "w") as f:
        json.dump(res, f, indent=4)
        f.close()