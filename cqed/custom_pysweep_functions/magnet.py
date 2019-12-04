# Authors: Lukas Splitthoff, Lukas Gruenhaupt @TUDelft
# 04-DEC-2019

''' A set of functions to conveniently use for magnet sweeping with pysweep.'''

from pysweep.core.measurementfunctions import MakeMeasurementFunction
from pysweep.core.sweepobject import SweepObject
from pysweep.databackends.base import DataParameter
import numpy as np


@MakeMeasurementFunction([DataParameter(name='x', unit='T'), 
                          DataParameter(name='y', unit='T'),
                          DataParameter(name='z', unit='T')])
def measure_magnet_components(d):
    ''' Measure the x, y, z component of the magnet and return as list
    output: [x, y, z]
    '''
    station = d['STATION']
    x_meas = station.mgnt.x_measured()
    y_meas = station.mgnt.y_measured()
    z_meas = station.mgnt.z_measured()
    return [x_meas, y_meas, z_meas] 


def sweep_phi(r, theta, points, max_field_strength=1.5):
    ''' Generate a pysweep.SweepObject to sweep phi at fixed amplitude and theta.
    Here we use the ISO 80000-2:2009 physics convention for the (r, theta, phi) <--> (x, y, z) definition. 
    Only values for the x, y, and z component are passed to the magnet.

    Inputs:
    r (float): magentic field strength, unit: T
    theta (float): inclination angle, unit: degrees, 0 <= theta <= 180
    points (float): sweep values for phi, i.e. the azimuth (in plane) angle, unit: degrees, range: 0 <= phi <= 360 
    max_field_strength (float): maximum magnetic field strength, unit: T

    Output:
    pysweep.SweepObject
    '''
    @MakeMeasurementFunction([])
    def point_function(d):
        return points, []
    
    @MakeMeasurementFunction([])
    def set_function(phi, d):

        assert max_field_strength>r>0., 'The field amplitude must not exceed {} and be lager than 0.' \
                                        ' Upper limit can be adjusted with kwarg: max_field_strength.' \
                                        ' Proceed with caution (Mu-metal shields do not appreciate high fields!)'.format(max_field_strength)
        assert 0.<=theta<=180., 'The inclination angle must be equal or lager than 0 and smaller or equal than 180. Change setting!'  
        assert 0.<=phi<=2*180., 'The azimuth angle must be equal or lager than 0 and smaller or equal than 360. Change setting!'  
        
        station = d['STATION']
        
        x = r*np.sin(np.radians(theta))*np.cos(np.radians(phi))
        y = r*np.sin(np.radians(theta))*np.sin(np.radians(phi))
        z = r*np.cos(np.radians(theta))        
        
        station.mgnt.x_target(x)
        station.mgnt.y_target(y)
        station.mgnt.z_target(z)

        station.mgnt.ramp(mode='safe')
        
        return []
    
    return SweepObject(set_function = set_function, unit = 'degrees', label = 'phi', point_function = point_function, dataparameter=None)


def sweep_theta(r, phi, points, max_field_strength=1.5):
    ''' Generate a pysweep.SweepObject to sweep theta at fixed amplitude and phi.
    Here we use the ISO 80000-2:2009 physics convention for the (r, theta, phi) <--> (x, y, z) definition. 
    Only values for the x, y, and z component are passed to the magnet.

    Inputs:
    r (float): magentic field strength, unit: T
    phi (float): the azimuth (in plane) angle, unit: degrees, range: 0 <= phi <= 360
    points (float): sweep values for theta, i.e. the inclination angle, unit: degrees, range: 0 <= phi <= 180
    max_field_strength (float): maximum magnetic field strength, unit: T

    Output:
    pysweep.SweepObject
    '''
    @MakeMeasurementFunction([])
    def point_function(d):
        return points, []
    
    @MakeMeasurementFunction([])
    def set_function(theta, d):
        # Here we use the ISO 80000-2:2009 physics convention for the (r, theta, phi) <--> (x, y, z) definition. 
        # Note that r is the radial distance, theta the inclination and phi the azimuth (in-plane) angle. 
        # For uniqueness, we restrict the parameter choice to r>=0, 0<= theta <= pi and 0<= phi <= 2pi. 
        # The units are: r in T, theta in degrees, phi in degrees. 
        
        assert max_field_strength>r>0., 'The field amplitude must not exceed {} and be lager than 0.' \
                                        ' Upper limit can be adjusted with kwarg: max_field_strength.' \
                                        ' Proceed with caution (Mu-metal shields do not appreciate high fields!)'.format(max_field_strength)
        assert 0.<=theta<=180., 'The inclination angle must be equal or lager than 0 and smaller or equal than 180. Change setting!'  
        assert 0.<=phi<=2*180., 'The azimuth angle must be equal or lager than 0 and smaller or equal than 360. Change setting!'  
     
        station = d['STATION'] 

        x = r*np.sin(np.radians(theta))*np.cos(np.radians(phi))
        y = r*np.sin(np.radians(theta))*np.sin(np.radians(phi))
        z = r*np.cos(np.radians(theta))        
        
        station.mgnt.x_target(x)
        station.mgnt.y_target(y)
        station.mgnt.z_target(z)

        station.mgnt.ramp(mode='safe')
        
        return []
    
    return SweepObject(set_function = set_function, unit = 'degrees', label = 'theta', point_function = point_function, dataparameter=None )


def sweep_r(phi, theta, points, max_field_strength=1.5):
    ''' Generate a pysweep.SweepObject to sweep field amplitude at fixed phi and theta.
    Here we use the ISO 80000-2:2009 physics convention for the (r, theta, phi) <--> (x, y, z) definition. 
    Only values for the x, y, and z component are passed to the magnet.

    Inputs:
    phi (float): the azimuth (in plane) angle, unit: degrees, range: 0 <= phi <= 360
    theta (float): inclination angle, unit: degrees, 0 <= theta <= 180
    points (float): sweep values for r, i.e. the magentic field strength, unit: T, range: 0 <= r <= max_field_strength
    max_field_strength (float): maximum magnetic field strength, unit: T

    Output:
    pysweep.SweepObject
    '''
    @MakeMeasurementFunction([])
    def point_function(d):
        return points, []
    
    @MakeMeasurementFunction([])
    def set_function(r, d):
        # Here we use the ISO 80000-2:2009 physics convention for the (r, theta, phi) <--> (x, y, z) definition. 
        # Note that r is the radial distance, theta the inclination and phi the azimuth (in-plane) angle. 
        # For uniqueness, we restrict the parameter choice to r>=0, 0<= theta <= pi and 0<= phi <= 2pi. 
        # The units are: r in T, theta in degrees, phi in degrees. 
        
        assert max_field_strength>r>0., 'The field amplitude must not exceed {} and be lager than 0.' \
                                        ' Upper limit can be adjusted with kwarg: max_field_strength.' \
                                        ' Proceed with caution (Mu-metal shields do not appreciate high fields!)'.format(max_field_strength)
        assert 0.<=theta<=180., 'The inclination angle must be equal or lager than 0 and smaller or equal than 180. Change setting!'  
        assert 0.<=phi<=2*180., 'The azimuth angle must be equal or lager than 0 and smaller or equal than 360. Change setting!'  
     
        station = d['STATION']
        
        x = r*np.sin(np.radians(theta))*np.cos(np.radians(phi))
        y = r*np.sin(np.radians(theta))*np.sin(np.radians(phi))
        z = r*np.cos(np.radians(theta))        
        
        station.mgnt.x_target(x)
        station.mgnt.y_target(y)
        station.mgnt.z_target(z)

        station.mgnt.ramp(mode='safe')
        
        return []
    
    return SweepObject(set_function = set_function, unit = "T", label = "r", point_function = point_function, dataparameter=None )