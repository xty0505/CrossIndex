import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from crossindex_main import CrossIndex

experiment_config = {
    "flights":{
        "dimensions": ["ARR_DELAY","DEP_DELAY","DISTANCE","AIR_TIME","ARR_TIME","DEP_TIME"],
        "types": ["numerical","numerical","numerical","numerical","numerical","numerical"],
        "bin_width": [5,5,200,20,1,1],
        "offset": [-20,-20,0,0,0,0],
        "cube_dir":"cube/experiment/Flights/"
    },
    "movies":{
        "dimensions": ["Running_Time_min","US_Gross","IMDB_Rating","Production_Budget","US_DVD_Sales","Rotten_Tomatoes_Rating","Worldwide_Gross","Release_Date"],
        "types": ["numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical"],
        "bin_width": [20.0,5.0,0.5,2.0,2.0,5.0,5.0,30541006451.612904],
        "offset": [0,0,0,0,0,0,0,315550800000],
        "cube_dir":"cube/experiment/Movies/"
    },
    "weather":{
        "dimensions": ["TEMP_MIN","TEMP_MAX","SNOW","ELEVATION","LONGITUDE","PRECIPITATION","WIND","RECORD_DATE"],
        "types": ["numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical"],
        "bin_width": [5.0,5.0,50.0,200.0,20.0,0.5,0.5,163725000.0],
        "offset": [-10,-10,0,-200,-160,0,0,1325307600000],
        "cube_dir":"cube/experiment/Weather/"
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
    datasets = ["flights", "movies", "weather"]
    for dataset in datasets:
        print('start dataset', dataset)
        config = experiment_config[dataset]
        print(config)
        sizes = ["10M"]
        for size in sizes:
            print('start building %s_%s'%(dataset,size))
            name = dataset+"_"+size
            # 基数从低到高
            crossindex = CrossIndex(name+"_"+"asc", config["dimensions"], config["types"])
            crossindex.adjust_by_cardinality(dataset_input_dir[dataset][size], config["bin_width"], config["offset"], ",", reverse=False)
            crossindex.build_csv()
            crossindex.save_csv(config)
            print('Ordered by ASC finished.')
            # 基数从高到低
            crossindex = CrossIndex(name+"_"+"desc", config["dimensions"], config["types"])
            crossindex.adjust_by_cardinality(dataset_input_dir[dataset][size], config["bin_width"], config["offset"], ",", reverse=True)
            crossindex.build_csv()
            crossindex.save_csv(config)
            print('Ordered by DESC finished.')
            # 随机
            crossindex = CrossIndex(name+"_"+"random", config["dimensions"], config["types"])
            crossindex.adjust_by_cardinality(dataset_input_dir[dataset][size], config["bin_width"], config["offset"], ",", rd=True)
            crossindex.build_csv()
            crossindex.save_csv(config)
            print('Ordered by random finished.')
        print('%s finished.'%dataset)
    
    print('Building crossindex by different order ends.')