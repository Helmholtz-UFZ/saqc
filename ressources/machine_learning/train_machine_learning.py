import pandas as pd
import numpy as np
import random #for random sampling of training/test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, classification_report
import joblib # for saving of model objects

###--------------------
### EXAMPLE PARAMETRIZATION:
###--------------------

# pd.options.mode.chained_assignment = None  # default='warn'
# data = pd.read_feather("data/sm/02_data.feather")
# data = data.reset_index()#data.index has to be reset as I use row nos only for indexing
#
# ### Reflagging
# index_manual = data.Flag == "Manual"
# data["FlagMan"] = index_manual.astype("int")# True/False as 0 or 1
# index_auto = data.Flag.str.contains("Auto")
# data["flag_bin"] = index_auto.astype("int")# True/False as 0 or 1
#
# field = "Target"
# references = ["Var1","Var2"]
# window_values = 20
# window_flags = 20
# modelname="testmodel"
# #groupvar = 0.2
# path = "saqc/ressources/machine_learning/models/"
# sensor_field="SensorID"
# group_field = "GroupVar"


def trainML(data, field, references, sensor_field:str, group_field:str, window_values:int = 20, window_flags:int = 20, path:str, modelname:str, testratio:float, **kwargs):

    """This Function trains machine-learning models to reproduce manual flags that were
    set for a specific variable. Inputs to the model are the timeseries of the
    respective target variable at multiple sensors, the automatic flags that were assigned by SaQC as well as multiple reference series.
    Internally, context information for each point is gathered in form of moving windows to improve the flagging algorithm. By default, for both the target timeseries and the automatic flags, the
    information of the previous and preceeding timestep are gathered. Next, according to user inputs of window_flags and window_values, the number of flags
    and the mean gradient of the specified windows is calculated, both for t+windowsize and t-windowsize. The moving window calculations are executed for each sensor, seperately,
    and multiple models are trained, one for each level a grouping variable that can be defined by the user. The model objects that can be used for future flagging are stored
    along with log-files that store the models`accuracy on training and test.


    :param data:                        The pandas dataframe holding the data of the target variable at multiple sensors in long format, i.e. concatenated row-wise.
                                        Along with this, there should be columns with the respective series of reference variables and a column of quality flags. The latter
                                        should contain both automatic and manual flags.
    :param field:                       Fieldname of the field in data that is to be flagged
    :param references:                  A list of strings, denoting the fieldnames of the data series that should be used as reference variables
    :parameters sensor_field:           A string denoting the fieldname of unique sensor-IDs
    :parameter group_field:             A string denoting the fieldname of the grouping variable. For each level of this variable, a seperate model will be trained.
    :param window_values:               An integer, denoting the window size that is used to derive the gradients of both the field- and reference-series inside the moving window
    :param window_flags:                An integer, denoting the window size that is used to count the surrounding automatic flags that have been set before
    :param path:                        A string denoting the path to the folder where the model objects along with log-files should be saved to
    :param modelname:                   A string denoting the name of the model. The name is used for naming of the model objects as well as log-files. Naming will
                                        be: 'modelname'_'value of group_field'.pkl
    :param testratio                    A float denoting the ratio of the test- vs. training-set to be drawn from the data, e.g. 0.3
    """

    randomseed = 36
    ### Prepare data, i.e. compute moving windows
    print("Computing time-lags")
    # save original row index for merging into original dataframe, as NAs will be introduced
    data = data.rename(columns={"index":"RowIndex"})
    # define Test/Training
    data = data.assign(TeTr = "Tr")
    # create empty df for training data
    traindata = pd.DataFrame()
    # calculate windows
    for sensor_id in data[sensor_field].unique():
        print(sensor_id)
        sensordf = data.loc[data[sensor_field]==sensor_id]
        index_test = sensordf.RowIndex.sample(n=int(testratio*len(sensordf)), random_state=randomseed)#draw random sample
        sensordf.TeTr[index_test] = "Te"#assign test samples

        sensordf["flag_bin_t_1"] = sensordf["flag_bin"]-sensordf["flag_bin"].shift(1)# Flag at t-1
        sensordf["flag_bin_t1"] = sensordf["flag_bin"]-sensordf["flag_bin"].shift(-1)# Flag at t+1
        sensordf["flag_bin_t_"+str(window_flags)] = sensordf["flag_bin"].rolling(window_flags+1,center=False).sum()# n Flags in interval t to t-window_flags
        sensordf["flag_bin_t"+str(window_flags)] = sensordf["flag_bin"].iloc[::-1].rolling(window_flags+1,center=False).sum()[::-1]# n Flags in interval t to t+window_flags
        # forward-orientation not possible, so right-orientation on reversed data an reverse result

        # Add context information for field+references
        for i in [field]+references:
            sensordf = pd.concat([sensordf,refCalc(reference=sensordf[i],window_values=window_values)],axis=1)

        # write back into new dataframe
        traindata = traindata.append(sensordf)

    # remove rows that contain NAs (new ones occured during predictor calculation)
    traindata = traindata.dropna(axis=0,how="any")


    ################
    ### FIT Model
    ################
    n_cores = os.getenv('NSLOTS', 1)
    print("MODEL TRAINING ON "+str(n_cores)+" CORES")

    # make column in "traindata" to store predictions
    traindata = traindata.assign(PredMan=0)
    outinfo_df = []
    resultfile = open(os.path.join(path,modelname+"_resultfile.txt"),"w")
    starttime = time.time()
    # For each category of groupvar, fit a separate model

    for groupvar in traindata[group_field].unique():
        resultfile.write("GROUPVAR: " + str(groupvar)+"\n")
        print("GROUPVAR: " + str(groupvar))
        print("TRAINING MODEL...")
        # drop unneeded columns
        groupdata = traindata[traindata[group_field]==groupvar].drop(columns=["Time","RowIndex","Flag","flag_bin","PredMan",group_field, sensor_field])
        forest = RandomForestClassifier(n_estimators=500, random_state=randomseed,oob_score=True,n_jobs=-1)
        X_tr = groupdata.drop(columns=["TeTr","FlagMan"])[groupdata.TeTr=="Tr"]
        Y_tr = groupdata.FlagMan[groupdata.TeTr=="Tr"]
        forest.fit(y=Y_tr, X=X_tr)
        # save model object
        joblib.dump(forest, os.path.join(path,modelname+"_"+str(groupvar)+".pkl"))
        # retrieve training predictions
        print("PREDICTING...")
        preds_tr = forest.oob_decision_function_[:,1]>forest.oob_decision_function_[:,0]#training, derive from OOB class votes
        preds_tr = preds_tr.astype("int")

        # get test predictions
        X_te = groupdata.drop(columns=["TeTr","FlagMan"])[groupdata.TeTr=="Te"]
        Y_te = groupdata.FlagMan[groupdata.TeTr=="Te"]
        preds_te = forest.predict(X_te)#test

        # Collect info on model run (n datapoints, share of flags, Test/Training accuracy...)
        outinfo = [groupvar,groupdata.shape[0],len(preds_tr),len(preds_te),sum(groupdata.FlagMan[groupdata.TeTr=="Tr"])/len(preds_tr)*100,sum(groupdata.FlagMan[groupdata.TeTr=="Te"])/len(preds_te)*100,\
                   recall_score(Y_tr,preds_tr),recall_score(Y_te,preds_te),\
                   precision_score(Y_tr,preds_tr),precision_score(Y_te,preds_te)]
        resultfile.write("TRAINING RECALL:"+"\n")
        resultfile.write(str(recall_score(groupdata.FlagMan[groupdata.TeTr=="Tr"],preds_tr))+"\n")# Training error (Out-of-Bag)
        resultfile.write("TEST RECALL:"+"\n")
        resultfile.write(str(recall_score(groupdata.FlagMan[groupdata.TeTr=="Te"],preds_te))+"\n"+"\n")# Test error
        outinfo_df.append(outinfo)
        # save back to dataframe
        traindata.PredMan[(traindata.TeTr=="Tr") & (traindata[group_field]==groupvar)] = preds_tr
        traindata.PredMan[(traindata.TeTr=="Te") & (traindata[group_field]==groupvar)] = preds_te

    endtime = time.time()
    print("TIME ELAPSED: "+str(timedelta(seconds=endtime-starttime))+" min")
    outinfo_df = pd.DataFrame.from_records(outinfo_df,columns=[group_field,"n","n_Tr", "n_Te", "Percent_Flags_Tr","Percent_Flags_Te", "Recall_Tr", "Recall_Te", "Precision_Tr","Precision_Te"])
    outinfo_df = outinfo_df.assign(Modelname=modelname)
    resultfile.write(str(outinfo_df))
    outinfo_df.to_csv(os.path.join(path,modelname+"_outinfo.csv"),index=False)
    resultfile.close()

    # write results back into original "data" dataframe
    data = data.assign(PredMan=np.nan)
    data.PredMan[traindata.RowIndex] = traindata.PredMan# based on RowIndex as NAs were created in traindata
    data.to_feather("data/sm/03_data_preds")

trainML(data,field, references, sensor_field,group_field, window_values, window_flags, path, modelname,0.3)
