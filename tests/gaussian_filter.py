import numpy as np

PROB = 0.00001
ODDS = 1.4

def main():
    gfilter = apply_gaussian_filter(ODDS,PROB)
    impliededge = (ODDS - 1/PROB) 
    w = (ODDS-1.0)*np.log(ODDS)/2
    print("if implied edge > w then supression begins")
    print({"odds": ODDS, "prob": PROB, "gfilter": gfilter, "implied edge": impliededge, "w": w})

def apply_gaussian_filter(odds, prob):
    ##sigma here is set in such a way that as we deviate from 2.0 odds the drop off of the gaussian increases with the square
    sigma = np.log(1/(PROB*PROB))

    w = (odds-1.0)*np.log(odds)/2
    diff = abs(odds - 1/prob)

    #plateaued curve.
    exp_component = 1.0 if diff<=w else np.exp(-np.power(diff,2)/(2*np.power(sigma,2)))

    ##return
    return exp_component

if __name__ == "__main__":
    main()
