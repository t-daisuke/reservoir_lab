"""
Reservoir Autoencoder
version 0.6
(c)2021 Junya Kato
"""




def autoreservoir(seed,
                  teacher,
                  testing_data,
                  testing_data_abnormal,
                  NFFT,
                  noverlap,
                  bias,
                  inputScaling,
                  teacherScaling,
                  teacherShift,
                  InputDimension,
                  OutputDimension,
                  ReservoirSize,
                  Connectivity,
                  SpectralRadius,
                  noise,
                  LeakingRate,
                  delay,
                  dir):

    import numpy as np
    import matplotlib.pyplot as plt
    import soundfile as sf
    import os
    import pandas as pd
    import gc
    np.random.seed(seed)

    # function for normalization, mean 0, std 1
    def zscore(x):
        xmean = x.mean()
        xstd = np.std(x)
        z = (x - xmean)/xstd
        return z

    # function for generating random matrix
    # m x n matrix with certain density
    # nonzero values -0.5~0.5
    def sparserand(m, n, density):
      matrix = np.zeros([m,n])
      place = []
      count = 0
      limit=round(m*n*density)
      while count<limit:
        try_index=[np.random.randint(0,m), np.random.randint(0,n)]
        if try_index not in place:
          place.append(try_index)
          count=count+1
      for index in place:
        matrix[index[0], index[1]]=np.random.random(1) - 0.5
      return matrix

    def delay_training_inputsignal(input_scaled, delay):
        if delay > 0:
            return np.delete(input_scaled, np.s_[:delay], axis=1)
        elif delay < 0:
            return np.delete(input_scaled, np.s_[delay:], axis=1)
        else:
            return input_scaled

    def delay_training_teachersignal(output_scaled, delay):
        rolled = np.roll(output_scaled, delay, axis=1)
        if delay > 0:
            return np.delete(rolled, np.s_[:delay], axis=1)
        elif delay < 0:
            return np.delete(rolled, np.s_[delay:], axis=1)
        else:
            return rolled

    # give scaled testing data and get scaled output
    def process_reservoir(testingInputScaled, InitialState):
        ## import last input from training process
        ## testingInputScaled = np.hstack([inputScaled[:, -1].reshape(InputDimension, 1), testingInputScaled])
        # X as reservoir state when predicting
        ## import last state from training process
        # X = np.hstack([XX[:, -1].reshape(ReservoirSize, 1), np.zeros((ReservoirSize, np.shape(testingInputScaled)[1]))])
        X = np.zeros((ReservoirSize, np.shape(testingInputScaled)[1]))
        print('shape of X: {0}'.format(np.shape(X)))
        ## import last output from training process
        ## testingOutput = np.hstack([teacherScaled[:, -1].reshape(OutputDimension, 1), np.zeros((OutputDimension, np.shape(testingInputScaled)[1]))])
        testingOutput = np.zeros((OutputDimension, np.shape(testingInputScaled)[1]))
        # computing X and testingOutput
        # x as single reservoir state
        x = InitialState
        # x = np.zeros((ReservoirSize, 1))
        # x = XX[:, -1].reshape(ReservoirSize, 1)
        for t in range(np.shape(testingInputScaled)[1]):
            u = testingInputScaled[:, t].reshape(InputDimension, 1)
            x = (1 - LeakingRate) * x + LeakingRate * np.tanh(np.dot(Win, u) + np.dot(W, x)) + (
                        noise * (np.random.rand(ReservoirSize, 1) - 0.5))
            X[:, t] = x[:, 0]
            testingOutput[:, t] = np.tanh(np.dot(Wout, x[:, 0]))
        return testingOutput, X

    def calc_rmse(output, teacher):
        error = output - teacher
        error_squared = error ** 2
        rmse_series = []
        for n in range(np.shape(error_squared)[1]):
            rmse = (np.mean(error_squared[:, n])) ** 0.5
            rmse_series.append(rmse)
        return rmse_series

    def input_preprocess_calc_rmse(input, delay):
        if delay > 0:
            return np.delete(input, np.s_[-1 * delay:], axis=1)
        elif delay < 0:
            return np.delete(input, np.s_[:-1 * delay], axis=1)
        else:
            return input

    def output_preprocess_calc_rmse(output, delay):
        if delay > 0:
            return np.delete(output, np.s_[:delay], axis=1)
        elif delay < 0:
            return np.delete(output, np.s_[delay:], axis=1)
        else:
            return output



    # arranging training input, column vector as signal
    wav, fs = sf.read(teacher)
    wav = wav[:, 0]
    # normalization mean 0, std 1
    wav = zscore(wav)
    # FFT
    fig = plt.figure()
    pxx, freqs, time, im = plt.specgram(wav, NFFT=NFFT,
                                        Fs=fs, window=plt.mlab.window_hanning,
                                        noverlap=noverlap, cmap='jet', mode='magnitude')
    fig.savefig("teacher.png")
    # Setting input and teacher
    inputScaled_pre = np.vstack([bias * np.ones((1, np.shape(pxx)[1])),
                             inputScaling * pxx])
    teacherScaled_pre = (pxx * teacherScaling) + teacherShift
    print('shape of teacherScaled: {0}'.format(np.shape(teacherScaled_pre)))
    # Preprocess for delay data
    inputScaled = delay_training_inputsignal(inputScaled_pre, delay)
    teacherScaled = delay_training_teachersignal(teacherScaled_pre, delay)

    # Setting input weight -1~1, uniform distribution
    Win = np.random.rand(ReservoirSize, InputDimension) * 2 - 1
    # Start setting reservoir weight
    W = sparserand(ReservoirSize, ReservoirSize, Connectivity)
    # Compute eigenvalues w and eigenvectors v
    w, v = np.linalg.eig(W)
    # print("w: {0}".format(w))
    # Setting Spectral radius
    W = W * SpectralRadius / max(abs(w))

    # training Wout
    # xx as single reservoir state, XX as reservoir states matrix when training
    XX = np.zeros((ReservoirSize, np.shape(inputScaled)[1]))
    print('shape of XX: {0}'.format(np.shape(XX)))
    xx = np.zeros((ReservoirSize, 1))
    print('shape of xx: {0}'.format(np.shape(xx)))
    for t in range(np.shape(inputScaled)[1]):
        # need to reshape otherwise shape of u would be (InputDimension, )
        # We want (InputDimension, 1)
        # Be smart, Python!!!
        u = inputScaled[:, t].reshape(InputDimension, 1)
        xx = (1 - LeakingRate)*xx + LeakingRate*np.tanh(np.dot(Win, u) + np.dot(W, xx)) + (noise * (np.random.rand(ReservoirSize, 1) - 0.5))
        XX[:, t] = xx[:, 0]
    allX = XX
    # ignore initial state
    transient = 100
    XX = XX[:, transient:]
    # compute Wout
    Wout = np.dot(np.arctanh(teacherScaled[:, transient:]), np.linalg.pinv(XX))
    print("Training Done!")

    #######################################################
    #
    # prediction using normal data
    # arranging testing input, column vector as signal
    #
    #######################################################
    wav, fs = sf.read(testing_data)
    wav = wav[:, 0]
    # normalization mean 0, std 1
    wav = zscore(wav)
    # FFT
    fig = plt.figure()
    pxx_test, freqs, time, im = plt.specgram(wav, NFFT=NFFT,
                                        Fs=fs, window=plt.mlab.window_hanning,
                                        noverlap=noverlap, cmap='jet', mode='magnitude')
    fig.savefig("test_normal.png")

    # Setting input scaling and bias
    testingInputScaled = np.vstack([bias * np.ones((1, np.shape(pxx_test)[1])),
                             inputScaling * pxx_test])
    testingOutput, X =process_reservoir(testingInputScaled, InitialState=np.zeros((ReservoirSize, 1)))
    '''
    ## import last input from training process
    ## testingInputScaled = np.hstack([inputScaled[:, -1].reshape(InputDimension, 1), testingInputScaled])
    # X as reservoir state when predicting
    ## import last state from training process
    # X = np.hstack([XX[:, -1].reshape(ReservoirSize, 1), np.zeros((ReservoirSize, np.shape(testingInputScaled)[1]))])
    X = np.zeros((ReservoirSize, np.shape(testingInputScaled)[1]))
    print('shape of X: {0}'.format(np.shape(X)))
    ## import last output from training process
    ## testingOutput = np.hstack([teacherScaled[:, -1].reshape(OutputDimension, 1), np.zeros((OutputDimension, np.shape(testingInputScaled)[1]))])
    testingOutput = np.zeros((OutputDimension, np.shape(testingInputScaled)[1]))
    # computing X and testingOutput
    # x as single reservoir state
    x = np.zeros((ReservoirSize, 1))
    # x = XX[:, -1].reshape(ReservoirSize, 1)
    for t in range(np.shape(testingInputScaled)[1]):
        u = testingInputScaled[:, t].reshape(InputDimension, 1)
        x = (1 - LeakingRate)*x + LeakingRate*np.tanh(np.dot(Win, u) + np.dot(W, x)) + (noise * (np.random.rand(ReservoirSize, 1) - 0.5))
        X[:, t] = x[:, 0]
        testingOutput[:, t] = np.tanh(np.dot(Wout, x[:, 0]))
    '''
    # unscale testingOutput
    testingOutput = (testingOutput-teacherShift)/teacherScaling
    print("normal output done!")

    # compute RMSE
    # preprocess before calculating rmse
    pxx_test_del = input_preprocess_calc_rmse(pxx_test, delay)
    testingOutput_del = output_preprocess_calc_rmse(testingOutput, delay)
    rmse_series = calc_rmse(testingOutput_del, pxx_test_del)



    # testing output for coding
    # print('pxx: {0}'.format(pxx))
    # print('shape of Win: {0}'.format(np.shape(Win)))
    # print('shape of inputScaled: {0}'.format(np.shape(inputScaled)))
    # print('shape of W: {0}'.format(np.shape(W)))
    # print('shape of x: {0}'.format(np.shape(x)))
    # print('shape of XX: {0}'.format(np.shape(XX)))
    # print('shape of Wout: {0}'.format(np.shape(Wout)))
    # print('shape of testingInputScaled: {0}'.format(np.shape(testingInputScaled)))
    # print('shape of X: {0}'.format(np.shape(X)))
    # print('shape of testingOutput: {0}'.format(np.shape(testingOutput)))
    # print('inputScaled: {0}'.format(inputScaled))
    # print('InputDimension: {0}'.format(InputDimension))
    # print('OutputDimension: {0}'.format(OutputDimension))
    # print('EsnSize: {0}'.format(ReservoirSize))
    # print('SpectralRadius: {0}'.format(SpectralRadius))
    # print('Win: {0}'.format(Win))
    # print('W: {0}'.format(Win_2))
    # print('W: {0}'.format(W))
    # print('w: {0}'.format(w))
    # print('v: {0}'.format(v))

    output_parent_folder = dir + "/LeakingRate" + str(LeakingRate)
    os.makedirs(output_parent_folder, exist_ok=True)
    output_folder = output_parent_folder + "/delay" + str(delay)
    os.mkdir(output_folder)

    np.savetxt(output_folder + '/Wout.csv', Wout, delimiter=',')
    np.savetxt(output_folder + '/Win.csv', Win, delimiter=',')
    np.savetxt(output_folder + '/W.csv', W, delimiter=',')
    np.savetxt(output_folder + '/rmse_normal.csv', rmse_series, delimiter=',')

    # plot RMSE
    plt.figure()
    time = np.arange(0, len(rmse_series))/fs
    # plt.plot(time[:np.shape(rmse_series)[0]], rmse_series)
    plt.plot(time, rmse_series)
    # plt.ylim(0, 0.40)
    plt.ylim(0, 0.80)
    plt.savefig(output_folder + '/rmse.png')
    # plt.show()

    plt.figure()
    plt.imshow(Win, cmap='coolwarm')
    plt.colorbar()
    plt.savefig(output_folder + '/colormap_Win.png')
    # plt.show()

    # plt.imsave('colormap_W.png', W)
    plt.figure()
    plt.imshow(W, cmap='coolwarm')
    plt.colorbar()
    plt.savefig(output_folder + '/colormap_W.png')
    # plt.show()

    plt.figure()
    plt.imshow(XX[:, 80000:82000], cmap='coolwarm')
    plt.colorbar()
    plt.savefig(output_folder + '/colormap_XX.png')
    # plt.show()

    plt.figure()
    plt.imshow(Wout, cmap='coolwarm')
    plt.colorbar(orientation='horizontal')
    plt.savefig(output_folder + '/colormap_Wout.png')
    # plt.show()

    plt.figure()
    plt.imshow(X[:, 80000:82000], cmap='coolwarm')
    plt.colorbar()
    plt.savefig(output_folder + '/colormap_X.png')
    # plt.show()

    plt.figure(figsize=(6.0, 1.6))
    plt.imshow(Wout, cmap='coolwarm')
    plt.colorbar(aspect=10, shrink=0.50)
    plt.clim(-0.005, 0.005)
    plt.title('Wout(-0.005,0.005)')
    plt.savefig(output_folder + '/colormap_inspecting_Wout(-0.005,0.005).png')
    # plt.show()

    plt.figure(figsize=(6.0, 1.6))
    plt.imshow(Wout, cmap='coolwarm')
    plt.colorbar(aspect= 10, shrink=0.50)
    plt.clim(-0.01, 0.01)
    plt.title('Wout(-0.01,0.01)')
    plt.savefig(output_folder + '/colormap_inspecting_Wout(-0.01,0.01).png')
    # plt.show()

    plt.figure(figsize=(6.0, 1.6))
    plt.imshow(Wout, cmap='coolwarm')
    plt.colorbar(aspect=10, shrink=0.50)
    plt.clim(-0.05, 0.05)
    plt.title('Wout(-0.05,0.05)')
    plt.savefig(output_folder + '/colormap_inspecting_Wout(-0.05,0.05).png')
    # plt.show()

    plt.figure(figsize=(6.0, 1.6))
    plt.imshow(Wout, cmap='coolwarm')
    plt.colorbar(aspect=10, shrink=0.50)
    plt.clim(-0.1, 0.1)
    plt.title('Wout(-0.1,0.1)')
    plt.savefig(output_folder + '/colormap_inspecting_Wout(-0.1,0.1).png')
    # plt.show()


    pd.Series(Wout.ravel()).describe().to_csv(output_folder + '/description.txt')

    ###########################################################
    #
    #
    # prediction using abnormal data
    #
    #
    ###########################################################

    wav, fs = sf.read(testing_data_abnormal)
    wav = wav[:, 0]
    # normalization mean 0, std 1
    wav = zscore(wav)
    # FFT
    fig = plt.figure()
    pxx_test_abnormal, freqs, time, im = plt.specgram(wav, NFFT=NFFT,
                                        Fs=fs, window=plt.mlab.window_hanning,
                                        noverlap=noverlap, cmap='jet', mode='magnitude')
    fig.savefig("test_abnormal.png")
    # concatenate teacher and abnormal data
    # pxx_test_abnormal_concatenate = np.hstack([pxx, pxx_test_abnormal])
    # Setting input scaling and bias
    testingInputScaled_abnormal = np.vstack([bias * np.ones((1, np.shape(pxx_test_abnormal)[1])),
                             inputScaling * pxx_test_abnormal])
    testingOutput, X_abnormal = process_reservoir(testingInputScaled_abnormal, InitialState=np.zeros((ReservoirSize, 1)))
    # unscale testingOutput
    testingOutput = (testingOutput-teacherShift)/teacherScaling

    # compute RMSE
    # preprocess before calculating rmse
    pxx_test_del = input_preprocess_calc_rmse(pxx_test_abnormal, delay)
    testingOutput_del = output_preprocess_calc_rmse(testingOutput, delay)
    rmse_abnormal_series = calc_rmse(testingOutput_del, pxx_test_del)

    np.savetxt(output_folder + '/rmse_abnormal.csv', rmse_abnormal_series, delimiter=',')

    # plot RMSE
    plt.figure()
    time = np.arange(0, len(rmse_abnormal_series)) / fs
    plt.plot(time, rmse_abnormal_series)
    # plt.ylim(0, 0.40)
    plt.ylim(0, 0.80)
    plt.savefig(output_folder + '/rmse_abnormal.png')
    plt.show()

    plt.figure()
    plt.imshow(X_abnormal[:, 80000:82000], cmap='coolwarm')
    plt.colorbar()
    plt.savefig(output_folder + '/colormap_X_abnormal.png')
    plt.show()

    # ################################################################
    # #
    # # prediction using both normal and abnormal data
    # #
    # ################################################################
    # # concatenate teacher and abnormal data
    # pxx_test_normal_abnormal_concatenate = np.hstack([pxx_test, pxx_test_abnormal])
    # # Setting input scaling and bias
    # testingInputScaled_normal_abnormal = np.vstack([bias * np.ones((1, np.shape(pxx_test_normal_abnormal_concatenate)[1])),
    #                                          inputScaling * pxx_test_normal_abnormal_concatenate])
    # testingOutput, X_normal_abnormal = process_reservoir(testingInputScaled_normal_abnormal,
    #                                               InitialState=np.zeros((ReservoirSize, 1)))
    # # unscale testingOutput
    # testingOutput = (testingOutput - teacherShift) / teacherScaling
    #
    # # compute RMSE
    # # preprocess before calculating rmse
    # pxx_test_del = input_preprocess_calc_rmse(pxx_test_normal_abnormal_concatenate, delay)
    # testingOutput_del = output_preprocess_calc_rmse(testingOutput, delay)
    # rmse_normal_abnormal_series = calc_rmse(testingOutput_del, pxx_test_del)
    #
    # np.savetxt(output_folder + '/rmse_normal_abnormal.csv', rmse_normal_abnormal_series, delimiter=',')
    #
    # # plot RMSE
    # plt.figure()
    # time = np.arange(0, len(rmse_normal_abnormal_series)) / fs
    # plt.plot(time, rmse_normal_abnormal_series)
    # # plt.ylim(0, 0.40)
    # plt.ylim(0, 0.80)
    # plt.savefig(output_folder + '/rmse_normal_abnormal.png')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(X_normal_abnormal[:, 80000:82000], cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig(output_folder + '/colormap_X_normal_abnormal.png')
    # plt.show()



    ################################################################
    #
    # Calculating and displaying thresholding and evaluation
    #
    ################################################################

    f = open(output_folder + '/F-measure.txt', 'x')
    f.writelines(["np.random.seed(", str(seed), ")", "\n"])
    f.writelines(["teacher : ", teacher, "\n"])
    f.writelines(["testing normal data : ", testing_data, "\n"])
    f.writelines(["testing abnormal data : ", testing_data_abnormal, "\n"])
    f.write("Setting FFT\n")
    f.writelines(["NFFT : ", str(NFFT), "\n"])
    f.writelines(["noverlap : ", str(noverlap), "\n"])
    f.write("Setting input scaling and bias\n")
    f.writelines(["bias : ", str(bias), "\n"])
    f.writelines(["inputScaling : ", str(inputScaling), "\n"])
    f.write("Setting teacher scaling and bias\n")
    f.writelines(["teacherScaling : ", str(teacherScaling), "\n"])
    f.writelines(["teacherShift : ", str(teacherShift), "\n"])
    f.write("Setting ESN parameters\n")
    f.writelines(["InputDimension : ", str(InputDimension), "\n"])
    f.writelines(["OutputDimension : ", str(OutputDimension), "\n"])
    f.writelines(["ReservoirSize : ", str(ReservoirSize), "\n"])
    f.writelines(["Connectivity : ", str(Connectivity), "\n"])
    f.writelines(["SpectralRadius : ", str(SpectralRadius), "\n"])
    f.writelines(["noise : ", str(noise), "\n"])
    f.writelines(["LeakingRate : ", str(LeakingRate), "\n"])
    f.write("\n")


    # finding 90 percentile rmse
    rmse_normal_sorted = sorted(rmse_series)
    thresholding_10 = rmse_normal_sorted[int(-1*0.1*len(rmse_series))]
    f.write("normal data\n")
    print("")
    print("thresholding and normal data")
    f.write("Threshold derived from 90 percentile rmse of normal data : ")
    f.write(str(thresholding_10))
    f.write("\n")
    print("Threshold derived from 90 percentile  rmse of normal data : {0}".format(thresholding_10))

    thresholding_5 = rmse_normal_sorted[int(-1*0.05*len(rmse_series))]
    f.write("Threshold derived from 95 percentile rmse of normal data : ")
    f.write(str(thresholding_5))
    f.write("\n")
    print("Threshold derived from 95 percentile  rmse of normal data : {0}".format(thresholding_5))

    f.write("number of normal data : ")
    f.write(str(len(rmse_series)))
    f.write("\n")
    print("number of normal data : {0}".format(len(rmse_series)))
    fp_10 = sum(x>thresholding_10 for x in rmse_series)
    f.writelines(["number of normal data over 90 percentile threshold : ", str((fp_10)), "\n"])
    print("number of normal data over 90 percentile threshold : {0}".format(fp_10))
    fp_5 = sum(x>thresholding_5 for x in rmse_series)
    f.writelines(["number of normal data over 95 percentile threshold : ", str((fp_5)), "\n"])
    print("number of normal data over 95 percentile threshold : {0}".format(fp_5))


    f.write("\n")
    f.write("abnormal data and thresholding\n")
    print("")
    print("thresholding and abnormal data")
    f.writelines(["number of abnormal data is ", str(len(rmse_abnormal_series)), "\n"])
    print("number of abnormal data is {0}".format(len(rmse_abnormal_series)))
    tp_10 = sum(x>thresholding_10 for x in rmse_abnormal_series)
    f.writelines(["number of abnormal data over 90 percentile threshold : ", str(tp_10), "\n"])
    print("number of abnormal data over 90 percentile threshold : {0}".format(tp_10))
    tp_5 = sum(x>thresholding_5 for x in rmse_abnormal_series)
    f.writelines(["number of abnormal data over 95 percentile threshold : ", str(tp_5), "\n"])
    print("number of abnormal data over 95 percentile threshold : {0}".format(tp_5))

    f.write("\n")
    print("")
    f.write("Evaluation of anomaly detection\n")
    print("Evaluation of anomaly detection")
    f.write("\n")
    print("")
    f.write("in case of 90 percentile thresholding\n")
    print("in case of 90 percentile thresholding")
    f.writelines(["True Positive : ", str(tp_10), "\n"])
    print("True Positive : {0}".format(tp_10))
    f.writelines(["False Positive : ", str(fp_10), "\n"])
    print("False Positive : {0}".format(fp_10))
    fn_10 = len(rmse_abnormal_series) - tp_10
    f.writelines(["False Negative : ", str(fn_10), "\n"])
    print("False Negative : {0}".format(fn_10))
    tn_10 = len(rmse_normal_sorted) - fp_10
    f.writelines(["True Negative : ", str(tn_10), "\n"])
    print("True Negative : {0}".format(tn_10))
    ac_10 = (tp_10 + tn_10) / (tp_10 + fp_10 + fn_10 + tn_10)
    f.writelines(["Accuracy = (TP+TN)/(TP+FP+FN+TN) : ", str(ac_10), "\n"])
    print("Accuracy = (TP+TN)/(TP+FP+FN+TN) : {0}".format(ac_10))
    pc_10 = tp_10 / (tp_10 + fp_10)
    f.writelines(["Precision = TP/(TP+FP) : ", str(pc_10), "\n"])
    print("Precision = TP/(TP+FP) : {0}".format(pc_10))
    rc_10 = tp_10 / (tp_10 + fn_10)
    f.writelines(["Recall = TP/(TP+FN) : ", str(rc_10), "\n"])
    print("Recall = TP/(TP+FN) : {0}".format(rc_10))
    fm_10 = (2 * pc_10 * rc_10)/(pc_10 + rc_10)
    f.writelines(["F-measure = (2*Precision*Recall)/(Precision+Recall) : ", str(fm_10), "\n"])
    print("F-measure = (2*Precision*Recall)/(Precision+Recall) : {0}".format(fm_10))

    f.write("\n")
    print("")
    f.write("in case of 95 percentile thresholding\n")
    print("in case of 95 percentile thresholding")
    f.writelines(["True Positive : ", str(tp_5), "\n"])
    print("True Positive : {0}".format(tp_5))
    f.writelines(["False Positive : ", str(fp_5), "\n"])
    print("False Positive : {0}".format(fp_5))
    fn_5 = len(rmse_abnormal_series) - tp_5
    f.writelines(["False Negative : ", str(fn_5), "\n"])
    print("False Negative : {0}".format(fn_5))
    tn_5 = len(rmse_series) - fp_5
    f.writelines(["True Negative : ", str(tn_5), "\n"])
    print("True Negative : {0}".format(tn_5))
    ac_5 = (tp_5 + tn_5) / (tp_5 + fp_5 + fn_5 + tn_5)
    f.writelines(["Accuracy = (TP+TN)/(TP+FP+FN+TN) : ", str(ac_5), "\n"])
    print("Accuracy = (TP+TN)/(TP+FP+FN+TN) : {0}".format(ac_5))
    pc_5 = tp_5 / (tp_5 + fp_5)
    f.writelines(["Precision = TP/(TP+FP) : ", str(pc_5), "\n"])
    print("Precision = TP/(TP+FP) : {0}".format(pc_5))
    rc_5 = tp_5 / (tp_5 + fn_5)
    f.writelines(["Recall = TP/(TP+FN) : ", str(rc_5), "\n"])
    print("Recall = TP/(TP+FN) : {0}".format(rc_5))
    fm_5 = (2 * pc_5 * rc_5)/(pc_5 + rc_5)
    f.writelines(["F-measure = (2*Precision*Recall)/(Precision+Recall) : ", str(fm_5), "\n"])
    print("F-measure = (2*Precision*Recall)/(Precision+Recall) : {0}".format(fm_5))

    f.close()
    gc.collect()
    '''

    ################################################################
    #
    #
    # vary Wout and see what happens
    #
    #
    ################################################################

    # Vary and substitue Wout and predict
    threshold = 0.05

    ############################################
    # In case of Wout over threshold otherwise 0
    ############################################
    Wout_over_threshold = Wout.copy()
    state_over = (abs(Wout_over_threshold) < threshold)
    Wout_over_threshold[state_over] = 0
    filename = output_folder + '/Wout_over' + str(threshold) + '.csv'
    np.savetxt(filename, Wout_over_threshold, delimiter=',')

    # prediction
    # arranging testing input; column vector as

    #############
    # normal case
    #############

    # X as reservoir state when predicting
    ## import last state from training process
    X = np.hstack([XX[:, -1].reshape(ReservoirSize, 1), np.zeros((ReservoirSize, np.shape(testingInputScaled)[1]))])

    testingOutput = np.zeros((OutputDimension, np.shape(testingInputScaled)[1]))
    # computing X and testingOutput
    # x as single reservoir state
    # x = np.zeros((ReservoirSize, 1))
    x = XX[:, -1].reshape(ReservoirSize, 1)
    for t in range(np.shape(testingInputScaled)[1]):
        u = testingInputScaled[:, t].reshape(InputDimension, 1)
        x = (1 - LeakingRate) * x + LeakingRate * np.tanh(np.dot(Win, u) + np.dot(W, x)) + (
                    noise * (np.random.rand(ReservoirSize, 1) - 0.5))
        X[:, t + 1] = x[:, 0]
        testingOutput[:, t] = np.tanh(np.dot(Wout_over_threshold, x[:, 0]))
    # unscale testingOutput
    testingOutput = (testingOutput - teacherShift) / teacherScaling

    # compute RMSE
    # preprocess before calculating rmse
    pxx_test_del = input_preprocess_calc_rmse(pxx_test, delay)
    testingOutput_del = output_preprocess_calc_rmse(testingOutput, delay)
    rmse_series = calc_rmse(testingOutput_del, pxx_test_del)


    # draw graph of RMSE with varied Wout
    plt.figure()
    time = np.arange(0, len(wav)) / fs
    plt.plot(time[:np.shape(rmse_series)[0]], rmse_series)
    plt.ylim(0, 0.80)
    figname = output_folder + '/Wout_over' + str(threshold) + '_normal.png'
    plt.savefig(figname)
    plt.show()

    #############
    # abnormal case
    #############

    # X as reservoir state when predicting
    ## import last state from training process
    X = np.hstack([XX[:, -1].reshape(ReservoirSize, 1), np.zeros((ReservoirSize, np.shape(testingInputScaled)[1]))])

    testingOutput = np.zeros((OutputDimension, np.shape(testingInputScaled)[1]))
    # computing X and testingOutput
    # x as single reservoir state
    # x = np.zeros((ReservoirSize, 1))
    x = XX[:, -1].reshape(ReservoirSize, 1)
    for t in range(np.shape(testingInputScaled_abnormal)[1]):
        u = testingInputScaled_abnormal[:, t].reshape(InputDimension, 1)
        x = (1 - LeakingRate) * x + LeakingRate * np.tanh(np.dot(Win, u) + np.dot(W, x)) + (
                noise * (np.random.rand(ReservoirSize, 1) - 0.5))
        X[:, t + 1] = x[:, 0]
        testingOutput[:, t] = np.tanh(np.dot(Wout_over_threshold, x[:, 0]))
    # unscale testingOutput
    testingOutput = (testingOutput - teacherShift) / teacherScaling

    # compute RMSE
    # preprocess before calculating rmse
    pxx_test_del = input_preprocess_calc_rmse(pxx_test_abnormal, delay)
    testingOutput_del = output_preprocess_calc_rmse(testingOutput, delay)
    rmse_series = calc_rmse(testingOutput_del, pxx_test_del)

    # draw graph of RMSE with varied Wout
    plt.figure()
    time = np.arange(0, len(wav)) / fs
    plt.plot(time[:np.shape(rmse_series)[0]], rmse_series)
    plt.ylim(0, 0.80)
    figname = output_folder + '/Wout_over' + str(threshold) + '_abnormal.png'
    plt.savefig(figname)
    plt.show()

    #############################################
    # In case of Wout below threshold otherwise 0
    #############################################
    Wout_below_threshold = Wout.copy()
    state_below = (abs(Wout_below_threshold) >= threshold)
    Wout_below_threshold[state_below] = 0
    filename = output_folder + '/Wout_below' + str(threshold) + '.csv'
    np.savetxt(filename, Wout_below_threshold, delimiter=',')

    # prediction
    # arranging testing input; column vector as signal

    #############
    # normal case
    #############

    # X as reservoir state when predicting
    ## import last state from training process
    X = np.hstack([XX[:, -1].reshape(ReservoirSize, 1), np.zeros((ReservoirSize, np.shape(testingInputScaled)[1]))])

    testingOutput = np.zeros((OutputDimension, np.shape(testingInputScaled)[1]))
    # computing X and testingOutput
    # x as single reservoir state
    # x = np.zeros((ReservoirSize, 1))
    x = XX[:, -1].reshape(ReservoirSize, 1)
    for t in range(np.shape(testingInputScaled)[1]):
        u = testingInputScaled[:, t].reshape(InputDimension, 1)
        x = (1 - LeakingRate) * x + LeakingRate * np.tanh(np.dot(Win, u) + np.dot(W, x)) + (
                    noise * (np.random.rand(ReservoirSize, 1) - 0.5))
        X[:, t + 1] = x[:, 0]
        testingOutput[:, t] = np.tanh(np.dot(Wout_below_threshold, x[:, 0]))
    # unscale testingOutput
    testingOutput = (testingOutput - teacherShift) / teacherScaling

    # compute RMSE
    # preprocess before calculating rmse
    pxx_test_del = input_preprocess_calc_rmse(pxx_test, delay)
    testingOutput_del = output_preprocess_calc_rmse(testingOutput, delay)
    rmse_series = calc_rmse(testingOutput_del, pxx_test_del)

    # draw graph of RMSE with varied Wout
    plt.figure()
    time = np.arange(0, len(wav)) / fs
    plt.plot(time[:np.shape(rmse_series)[0]], rmse_series)
    plt.ylim(0, 0.80)
    figname = output_folder + '/Wout_below' + str(threshold) + '_normal.png'
    plt.savefig(figname)
    plt.show()

    ###############
    # abnormal case
    ###############

    # X as reservoir state when predicting
    ## import last state from training process
    X = np.hstack([XX[:, -1].reshape(ReservoirSize, 1), np.zeros((ReservoirSize, np.shape(testingInputScaled_abnormal)[1]))])

    testingOutput = np.zeros((OutputDimension, np.shape(testingInputScaled_abnormal)[1]))
    # computing X and testingOutput
    # x as single reservoir state
    # x = np.zeros((ReservoirSize, 1))
    x = XX[:, -1].reshape(ReservoirSize, 1)
    for t in range(np.shape(testingInputScaled_abnormal)[1]):
        u = testingInputScaled_abnormal[:, t].reshape(InputDimension, 1)
        x = (1 - LeakingRate) * x + LeakingRate * np.tanh(np.dot(Win, u) + np.dot(W, x)) + (
                noise * (np.random.rand(ReservoirSize, 1) - 0.5))
        X[:, t + 1] = x[:, 0]
        testingOutput[:, t] = np.tanh(np.dot(Wout_below_threshold, x[:, 0]))
    # unscale testingOutput
    testingOutput = (testingOutput - teacherShift) / teacherScaling

    # compute RMSE
    # preprocess before calculating rmse
    pxx_test_del = input_preprocess_calc_rmse(pxx_test_abnormal, delay)
    testingOutput_del = output_preprocess_calc_rmse(testingOutput, delay)
    rmse_series = calc_rmse(testingOutput_del, pxx_test_del)

    # draw graph of RMSE with varied Wout
    plt.figure()
    time = np.arange(0, len(wav)) / fs
    plt.plot(time[:np.shape(rmse_series)[0]], rmse_series)
    plt.ylim(0, 0.80)
    figname = output_folder + '/Wout_below' + str(threshold) + '_abnormal.png'
    plt.savefig(figname)
    plt.show()
    '''


    ########################################################
    # Do not erase return
    ########################################################
    return (fm_10, fm_5)
