# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:40:07 2025

@author: f0058kf
"""


#%% cox and johnson processing
def Cox_and_Johnson(cal_data, data, A, B, startTime, finishTime):
    ''' 
    Imports data from the VWSG logger and calculates primary/secondary stress based upon 
    biaxial strain measurments
    
    The data collected by the VWSG is NOT in principal stress space. The sensor calculates
    strain of each of the 3 rosette wires within the VWSG, which can be used to derive stress state. 
    During each sample the wire rosette is magnetically plucked and the freq^2 of each wire 
    is recorded. Primary and secondary principal stresses can be calculated using the 
    frequency reading, known material qualities of the sensor itself, and the inclusion factors
    of the sensor to the ice. Cox and Johnson 1983 outlines a process that converts these frequency 
    measurements to stresses. This function directly follows their technique.
    
    An important element to this function is the zeroing out of strain. V1off, V2off, and V3off are 
    variables created to save the wire strain state at a specified 'offTime', which is manually set 
    in the main loop. This is critical due to the freeze-in process causing anisotropic deformation
    of the cylinder. The uncorrected result is stress measurements that are completely biased to one wire. 
    If one wire is significantly more deformed in the freeze-in process the coordinate system cannot
    rotate to the direction of primary principal stress. This causes highly inaccurate stress states.
    One can eliminate this problem by subtracting the wire strain offset after freeze-in. 
    
    REFERENCE: "Stress Measurements in Ice" - Gordon Cox and Jerome Johnson, 1983
    
    ''' 
    
    # 1. Extract G values (modulus of rigidity) from calfile, provided in * 10^-6 
    
    # formula -> G = 4 * (p* l^2 / E)
    # where:
        # p = wire density
        # l = wire length (undeformed)
        # E = elastic modulus of the wire

    # convert from values provided, which are inches * 10^-6/digit
    # to m/digit by multiplying by 10E-6 and dividing by 39.4 in/m
    
    # 4. Filter data to time range of interest (set in main loop)
    data = data.loc[(data.index > startTime)]
    data = data.loc[(data.index < finishTime)]
    
    G1 = cal_data['G1'].iloc[0]*1E-6 # Modululs of rigidity along each axis
    G2 = cal_data['G2'].iloc[0]*1E-6 
    G3 = cal_data['G3'].iloc[0]*1E-6

    # 2. Calculate deformation of the wire (strain)
    # Deformation(m) = G(m/digit)*[zero_read_1(digit, units freq^2*10^-3) - reading(digit, units freq^2*10^-3)]
    # radial deformation of cylinder = lw/2*G*delta(Freq^2)
    # Therefore, formula -> V = G * f^2
    data = data.sort_index()
    data = data.loc[~data.index.duplicated(keep='first')]
    data['V1'] = G1 * np.array(data['Chan1_Digits_by_Stress'])*1/39.4 # radius of wire converted from inches to meters
    data['V2'] = G2 * np.array(data['Chan2_Digits_by_Stress'])*1/39.4
    data['V3'] = G3 * np.array(data['Chan3_Digits_by_Stress'])*1/39.4 
    
    V1 = data['V1']
    V2 = data['V2']
    V3 = data['V3']
    
    # 5. Determine principal and secondary stresses p and q 
    # For this, use equations 31 and 32 from Cox and Johnson 1983.
    # output is in Pa since units were converted earlier
    p = (1/2)*((1/(3*B))*((2*V1-V2-V3)**2 + 3*(V2-V3)**2)**0.5 + 1/(3*A)*(V1+V2+V3)) # maximum compressive principal stress (pg. 21)
    q = (1/(3*A))*(V1+V2+V3) - p    # the stress perpendicular to the maximum compressive principal stress 'p'

    np.seterr(divide='ignore', invalid='ignore')
    
    # equation 33 from Cox and Johnson 1983
    theta      = 0.5*np.arccos((V1-A*(p+q))/(B*(p-q))); 
    theta[theta < -np.pi] = theta[theta < -np.pi] + 2*np.pi
    AngleTest2 = A*(p+q) + B*(p-q)*np.cos(2*(theta+120*np.pi/180)); # If V2 = this, then theta is negative
    negative   = np.round(V2,4) == np.round(AngleTest2,4)           # Identifies which V2s are negative (per AngleTest 2, pg. 13)
    negative   = (negative-0.5)*2;                                  # Corrects angles to be on the range of [ -pi, pi ]
    theta      = negative * theta;

    theta[theta > np.pi/2]  = theta[theta > np.pi/2]  - np.pi;  # keep between pi and -pi
    theta[theta < -np.pi/2] = theta[theta < -np.pi/2] + np.pi;


    
    return(p, q, theta, data)


#%% temperature calibration

def tempCoeff(cal_data):
    ''' 
    Calculates the Temperature Correction Curve Coefficients as a Second Order Polynomial.
    This is done for each gauge using unique calibration curves.
    '''
    
    # Temperature calibrations are read in and a curve is fit to the data
    # y = (1/x) would fit curve better -> 07/21/21
    c1 = np.polyfit(cal_data['T'][0:4].astype(float),cal_data['Gage 1'][0:4].astype(float),2)
    c2 = np.polyfit(cal_data['T'][0:4].astype(float),cal_data['Gage 2'][0:4].astype(float),2)
    c3 = np.polyfit(cal_data['T'][0:4].astype(float),cal_data['Gage 3'][0:4].astype(float),2)
    return(c1, c2, c3)


def tempCorrection(data,coeffs1,coeffs2,coeffs3,cal_data):
    ''' 
    Determines temperature correction based upon calibration data and coefficients 
    '''
    
    # working in freq^2 *1E-3 units, first calculate the expected zero stress reading from the sensor at the current temperature 
    # use a second order polynomial fit to the temperature calibration data. Essentially this finds the offset of the dataset
    
    # First establish what the expected digits should be at this temperature if no load was applied
    data['Chan1_Digits_NoLoad_at_T'] = coeffs1[0]*data['Chan1_Temp'].astype(float)**2 + coeffs1[1]*data['Chan1_Temp'].astype(float) + coeffs1[2]
    # Now find the deviation between current reading and temp-corrected zero reading, Units are in freq^2 *1E-3
    # Subtracting the temperature offset from the digits reveals the specific stress digits
    # -data + offset to orient compression as positive, tension as negative (for what I can tell 07/21/21)
    data['Chan1_Digits_by_Stress']   = -data['Chan1_Digits'].astype(float) + data['Chan1_Digits_NoLoad_at_T']

    # Repeat for other channels, although temperature is only observed on channel 1.
    data['Chan2_Digits_NoLoad_at_T'] = coeffs2[0]*data['Chan1_Temp'].astype(float)**2+coeffs2[1]*data['Chan1_Temp'].astype(float)+coeffs2[2]
    data['Chan2_Digits_by_Stress']   = -data['Chan2_Digits'].astype(float)+data['Chan2_Digits_NoLoad_at_T']
    
    data['Chan3_Digits_NoLoad_at_T'] = coeffs3[0]*data['Chan1_Temp'].astype(float)**2+coeffs3[1]*data['Chan1_Temp'].astype(float)+coeffs3[2]
    data['Chan3_Digits_by_Stress']   = -data['Chan3_Digits'].astype(float)+data['Chan3_Digits_NoLoad_at_T']
    
    # Now Calculate the offset for each channel based upon the temperature coefficients and a preset test value
    T2 = 13.12 # hard coding for now to avoid issue of not having.
    Tcorr1 = (coeffs1[0]*T2**2 + coeffs1[1]*T2 + coeffs1[2]) - (coeffs1[0]*22.5**2 + coeffs1[1]*22.5 + coeffs1[2])
    Tcorr2 = (coeffs2[0]*T2**2 + coeffs2[1]*T2 + coeffs2[2]) - (coeffs2[0]*22.5**2 + coeffs2[1]*22.5 + coeffs2[2])
    Tcorr3 = (coeffs3[0]*T2**2 + coeffs3[1]*T2 + coeffs3[2]) - (coeffs3[0]*22.5**2 + coeffs3[1]*22.5 + coeffs3[2])
    
    # Stops setting multiple copies, removes nested problem
    pd.set_option('chained', None)
    
    # Implant the offset values within the calibration dataframe
    # cal_data['Offset1'][2] = Tcorr1
    # cal_data['Offset2'][2] = Tcorr2
    # cal_data['Offset3'][2] = Tcorr3
    
    # Use the offset to correct the wire strain readings directly
    data['Chan1_Digits_by_Stress'] = data['Chan1_Digits_by_Stress'] + Tcorr1# cal_data['Offset1'][2]
    data['Chan2_Digits_by_Stress'] = data['Chan2_Digits_by_Stress'] + Tcorr2#cal_data['Offset2'][2]
    data['Chan3_Digits_by_Stress'] = data['Chan3_Digits_by_Stress'] + Tcorr3#cal_data['Offset3'][2]
    
    # return the dataset
    return(data, cal_data)
