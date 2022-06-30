import matplotlib.pyplot as plt # To plot scatter graph
import pandas as pd # Use to read .csv file
import numpy as np #This is numerical library, used to simplify calculations.

Data = pd.read_csv("e:\HARSHIT\Anurag_Harshit\data.csv") # pd.read_csv function of panda used to read data from csv file
original_signal= Data["x[n]"].tolist() # real x[n]
disturbed_signal= Data["y[n]"].tolist() # blured and noised signal

def denoise(list):   # to denoise
    n =len(list)
    list=[list[0],list[0],list[0]]+ list+ [list[n-1],list[n-1],list[n-1]]   #At the edges we are padding the values to make kernals of same size  
    new_list=[]    # this will contain denoised signal
    for i in range(3,n+3):   # logic is to create a low pass filter manually using averaging method
        new_list.append((list[i-3]+list[i-2]+list[i-1]+list[i]+list[i+1]+list[i+2]+list[i+3])/7) # default averaging value is 7.
    return new_list  # returning denoised signal

# Discrete Time Fourier Transform of aperiodic signal      
def DTFT(x,unit=0.0009) :   # theoretically omega is continous but digitally it cannot be implemented hence we are using unit to increment it.
    # very small value of unit to eliminate spectral leakage
    Range_of_w=int(2*np.pi/unit)   # range of omega
    li=[]        #list to store DTFT
    for w in range (Range_of_w) :   # for loop for all the values of omega
        sum=0 
        for n in range (len(x)) :    # for loop for all the signal values for the variable signal x.
            sum+=x[n]*np.exp((-1)*1j*w*unit*n)   # Dtft formula using discrete values of w.
        li.append(sum)  # appending all the dtft values 
    return li  # returns dtft values as a list

def IDTFT(arr,unit= 0.0009): # inverse fourier transform
    Range_of_w= int(2*np.pi/unit)     
    li=[]  # list to store itft values
    for i in range(Range_of_w):    # for loop integration over a time period of 2pi.
        sum=0
        for j in range(Range_of_w):  #for loop for all values of omega
            sum+= arr[j] * np.exp(1j*i*unit*j)*unit/(2*np.pi)  #formula for inverse fourier transform using discrete values of w.
        li.append(sum) #append all itft values
    return li  # returning all itft values as a list
#deblurring - Logic is direct inverse filtering .
def deblur(blurred_signal,H):
    var=len(blurred_signal)  # H is impulse response: Output when input is delta[n]
    blurred_signal = [0,0] + blurred_signal #h[n] values in the question are defined from -2 to 2 but loop runs from 0 to 4 so ..                                
                                            # .. we add list [0,0] to shift 2 positions right.
    Y=DTFT(blurred_signal)  #DTFT of blurred singal             
    Hejw=DTFT(H)               #DTFT of impulse response 
    #Using convolution property in frequency domain we get that fourier transform of deblurred signal is fourier tansform of initial signal..
    #.. multiplied with fourier transform of impulse .So first we divide the transform of deblurred signal by transform of impulse reponse then..
    #..apply inverse dtft
    div=[]       #  Preparing list of values which have to be inversed 
    for i in range(len(Y)):
        # Doing Y[n]/H[n] only when Magnitude is greater than some random small value ..
        #..otherwise output will get amplified due to division by a very small number 
        if (Hejw[i].real**2+Hejw[i].imag**2) > 0.5:
            div.append(Y[i]/Hejw[i])    # this will give dtft X[i] which will be inversed to x[n] later
        else:
            div.append(Y[i])
    temp=IDTFT(div)  #assingment of ITFT of list to variable
    return temp[0:var]    #deblurring is only concerned upto values which are a part of the blurred signal so we use len function 
#Processing x1 and x2 to be compared for the conclusion.
h= [1/16,4/16,6/16,4/16,1/16] #given h[n] for blurring
x1 = deblur(denoise(original_signal),h)  #First case gives x1[n]
x2 = denoise(deblur(disturbed_signal,h))  #Second case gives x2[n]

#plt.plot for continuous and plt.scatter for discrete plots are derived from mathplotlib package 
#Arguments like size of dots,colour of dots can be assigned  with the values we need to plot and compare
#x_axis is the variable defined to show x axis having 193 values and the plot can compare x to x1 ,x2
#plt.show() is used to display all figures 
#plt.legend() assigns meaning to various plot elements

sample= [i for i in range(len(original_signal))]
plt.scatter(sample,original_signal,s =1,c='black')
plt.scatter(sample,x1,s=1,c='blue')
plt.scatter(sample,x2,s=1,c='orange')
plt.title("Comparison of x1[n] and x2[n] with x[n]")
plt.xlabel("Samples")
plt.ylabel("Signal values")
plt.legend(["x[n]","x1[n]","x2[n]"])
#plt.title("Comparison of x1[n] with x[n]")
#plt.legend(["x[n]","x1[n]"])
#plt.title("Comparison of x2[n] with x[n]")
#plt.legend(["x[n]","x2[n]"])
plt.show()

