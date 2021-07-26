import cqed.instrument_drivers.LeidenCryogenics.LCApplication as LCApp

# Connect to Temp Control and Front Panel applications

# For compiled executables:
TC = LCApp.LVApp("DRTempControl.Application","DR TempControl.exe\TC.vi")
FP = LCApp.LVApp("DRFrontPanel.Application","DR FrontPanel.exe\FP.vi")


# For VI's open inside LabVIEW IDE:
#TC = LCApp.LVApp("LabView.Application","E:\Dropbox\Labview\LC Software\TempControl\TC.vi") #Substitute with correct path to VI's
#FP = LCApp.LVApp("LabView.Application","E:\Dropbox\Labview\LC Software\Front Panel\FP.vi")

# List of currents for automatic mode
IList = ((0.0, 0.0, 10000.0, 7), (0.0, 0.0, 10000.0, 8.0), (0.0, 0.0, 10000.0, 9.0), (0.0, 0.0, 10000.0, 10.0),
         (0.0, 0.0, 10000.0, 11.0), (0.0, 0.0, 10000.0, 12.0), (0.0, 0.0, 10000.0, 14.0), (0.0, 0.0, 10000.0, 16.0),
         (0.0, 0.0, 10000.0, 18.0), (0.0, 0.0, 10000.0, 20.0), (0.0, 0.0, 10000.0, 25.0), (0.0, 0.0, 10000.0, 30.0),
         (0.0, 0.0, 10000.0, 35.0), (0.0, 0.0, 10000.0, 40.0), (0.0, 0.0, 10000.0, 50.0), (0.0, 0.0, 12000.0, 60.0),
         (0.0, 0.0, 12000.0, 70.0), (0.0, 0.0, 12000.0, 80.0), (0.0, 0.0, 12000.0, 90.0), (0.0, 0.0, 12000.0, 100.0),
         (0.0, 0.0, 14000.0, 110.0), (0.0, 0.0, 14000.0, 120.0), (0.0, 0.0, 14000.0, 140.0), (0.0, 0.0, 14000.0, 160.0),
         (0.0, 0.0, 15000.0, 200.0), (0.0, 0.0, 15000.0, 250.0), (0.0, 0.0, 15000.0, 300.0), (0.0, 0.0, 15000.0, 350.0),
         (0.0, 0.0, 15000.0, 400.0), (0.0, 0.0, 15000.0, 500.0), (0.0, 0.0, 15000.0, 600.0), (0.0, 0.0, 15000.0, 700.0),
         (0.0, 0.0, 10000.0, 0.0))


# Reading data from VI's
#NOTE: The names of the controld can vary between the software versions!
print(TC.GetData("Sensor Names")) #Channel names
print(TC.GetData("R Array")) #All Impedances
print(TC.GetData("R0")) #Particular Channel
print(TC.GetData("T Array")) #Temperatures
print(TC.GetData("T0")) #Particular Channel
print(TC.GetData("CS.Actual Current")) #Currents (mA) taking into account On/Off status
print(TC.GetData("I3")) #Current channel status (Current, On?, Gated?, Current flows?, Error?)


print(FP.GetData("GHS.ADC Data Array")) #FP pressures (P1-P7,Flow,Pcomp)
print(FP.GetData("MG.Data Array")) #MaxiGauge pressures
print(FP.GetData("PT1.Data Array")) #PT pressures and temperatures
print(FP.GetData("GHS.Layout1.LEDs")) #FP buttons state old layout
#print(FP.GetData("GHS.Layout2.LEDs")) #FP buttons state new layout

# Writing data - !!IMPORTANT!!: see LVApp.SetData() comments for details
TC.SetData("CS.I3.I",100) # Set 100 uA on CS channel 3
TC.SetData("CS.I3.Toggle ON/OFF",True) # Toggle CS channel 3 ON/OFF
TC.SetData("CS.Mode",1) #Switch to PID mode
TC.SetData("CS.Mode1.List",IList) # Write array to automatic mode currents list

FP.SetData("GHS.Button",58) # Toggle S4 pump ON/OFF


# It is possible to get names of all controls in the program:
#print(TC.GetData('ControlNames'))
#print FP.GetData('ControlNames')
# Control names starting with underscore ("_") are for internal use and should not be modified directly.
