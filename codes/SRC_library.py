import cvxpy as cvx
from numpy import linalg as LA
import numpy as np

import numpy.linalg
import scipy.io
import unittest
from pyCSalgos.BP.l1eq_pd import l1eq_pd

class SRC():

    # input will be np array in #(com_to_len,1216) #(com_to_len,380 ) (1216,) (380,)
    def run_src_classifier(self,data_train, data_test,label_train,label_test):
        _, samples_num = data_test.shape
        _, coef_num = data_train.shape
        A = data_train

        class_num = len(np.unique(label_train))

        right = 0  # predict ccorrect
        wrong = 0  # ~

        for test_ind in range(0, samples_num):
            print 'start predict sample ', test_ind
            # test_ind=2 #can try only one as example then comment for loop

            b = data_test.T[test_ind]


            print b.shape
            print type(b)
            print A.shape
            print type(A)

            # A * coef = b
            coef = cvx.Variable(coef_num)
            obj = cvx.Minimize(cvx.norm(coef, 1))
            const = [A * coef == b]
            prob = cvx.Problem(obj, const)
            result = prob.solve()
            print coef.value


            # x = cvx.Variable(coef_num)
            # gamma = cvx.Parameter(sign="positive")
            # error = cvx.sum_squares(A * x - b)
            # obj = cvx.Minimize(error)
            # const = [A * x == b]
            # prob = cvx.Problem(obj, const)
            # # prob = cvx.Problem(obj)
            # # print x.value

            # ------ after solved l1 --- get the class result--
            res = np.zeros(class_num)

            for i in range(1, class_num + 1):
                tempCoef = np.zeros(len(label_train))
                # find the index of that class
                index_of_class = np.where(label_train == i)[0]
                # put the coef's value into the tempCoef with correct index
                np.put(tempCoef, index_of_class, coef.value[index_of_class])
                temp_b = np.matmul(A, tempCoef)
                res[i - 1] = LA.norm(temp_b - b, 2)

            min_index = np.argmin(res) #et the class result
            # print min_index+1  #predicted label
            # print label_test[test_ind]  # true label

            # the predict class start from 0
            if (min_index + 1 == label_test[test_ind]):  # predict true
                right += 1
                print 'right'
            else:
                wrong += 1
                print 'wrong!'

        print 'total got: ', right + wrong
        print 'wrong is: ', wrong

        accuracy = float(right) / (right + wrong)
        print 'accuracy is: ', float(right) / (right + wrong)

        return accuracy, right, wrong


    def run_src_classifier_bf(self,data_train, data_test,label_train,label_test):
        _, samples_num = data_test.shape
        raw_dim, coef_num = data_train.shape
        A = data_train

        class_num = len(np.unique(label_train))

        right = 0  # predict ccorrect
        wrong = 0  # ~

        Y = data_test
        A = data_train


        mu, sigma = 0, 0.1
        ran_m = np.random.normal(mu, sigma, (200, raw_dim))
        # print ran_m.shape

        # A = np.matmul(ran_m, A)
        # Y = np.matmul(ran_m, Y)
        # print Y.shape
        # print A.shape

        # x0=A/y
        x0 = np.matmul(A.T, Y)
        # print x0
        print x0.shape

        for test_ind in range(0, samples_num):
            print 'start predict sample ', test_ind
            # test_ind=2 #can try only one as example then comment for loop

            xr = l1eq_pd(x0[:,test_ind], A, np.array([]), Y[:,test_ind])

            res = np.zeros(class_num)

            for i in range(1, class_num + 1):
                tempCoef = np.zeros(len(label_train))
                # find the index of that class
                index_of_class = np.where(label_train == i)[0]
                # put the coef's value into the tempCoef with correct index
                np.put(tempCoef, index_of_class, xr[index_of_class])
                temp_Y = np.matmul(A, tempCoef)
                res[i - 1] = LA.norm(temp_Y - Y[:,test_ind], 2)

                min_index = np.argmin(res)  # et the class result
                # print min_index+1  #predicted label
                # print label_test[test_ind]  # true label

                # the predict class start from 0
            if (min_index + 1 == label_test[test_ind]):  # predict true
                right += 1
                # print 'right'
            else:
                wrong += 1
                print 'wrong!'


        print 'total got: ', right + wrong
        print 'wrong is: ', wrong

        accuracy = float(right) / (right + wrong)
        print 'accuracy is: ', float(right) / (right + wrong)

        return accuracy, right, wrong