import numpy as np
import numpy.linalg as linalg

class Kalman:
    def __init__(self):
        pass
    
    def kf_predict(self, X, P, A, Q, B, U):
        X = np.dot(A, X) + np.dot(B, U)
        P = np.dot(A, np.dot(P, A.T)) + Q
        return (X, P)

    def kf_update(self, X, P, Y, H, R):
        IM = np.dot(H, X)
        IS = R + np.dot(H, np.dot(P, H.T))
        K = np.dot(P, np.dot(H.T, linalg.inv(IS)))
        X = X + np.dot(K, (Y - IM))
        P = P - np.dot(K, np.dot(IS, K.T))
        LH = self.gauss_pdf(Y, IM, IS)
        return (X, P, K, IM, IS, LH)

    def gauss_pdf(self, X, M, S):
        if M.shape[1] == 1:
            DX = X - np.tile(M, X.shape[1])
            E = 0.5 * np.sum(DX * (np.dot(linalg.inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(linalg.det(S))
            P = np.exp(-E)
        elif X.shape[1] == 1:
            DX = np.tile(X, M.shape[1]) - M
            E = 0.5 * np.sum(DX * (np.dot(linalg.inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(linalg.det(S))
            P = np.exp(-E)
        else:
            DX - X - M
            E = 0.5 * np.dot(DX.T, np.dot(linalg.inv(S), DX))
            E = E + 0.5 * M.shape[0] * np.log(2 * np.pi()) + 0.5 * np.log(linalg.det(S))
            P = np.exp(-E)
        
        return (P[0], E[0])