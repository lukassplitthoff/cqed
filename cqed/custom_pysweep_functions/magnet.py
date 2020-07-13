# Authors: Lukas Splitthoff, Lukas Gruenhaupt @TUDelft
# 04-DEC-2019

''' A set of functions to conveniently use for magnet sweeping with pysweep.'''

from pysweep.core.measurementfunctions import MakeMeasurementFunction
from pysweep.core.sweepobject import SweepObject
from pysweep.databackends.base import DataParameter
import numpy as np
from warnings import showwarning
from cqed.custom_pysweep_functions.field_aligner import FieldAligner
from qcodes import station

def field_limit(Bx, By, Bz, max_field_strength=1.5) -> bool:
    ''' Calculate the absolute field amplitude and check against the max_field_strength.
    For use with QCoDeS oxford.MercuryiPS_VISA driver.
    Inputs:
    Bx, By, Bz (float): magnetic field components
    Outputs:
    bool: true, if sqrt(Bx^2+By^2+Bz^2) < max_field_strength; else false
    '''
    if max_field_strength > 1.5:
        showwarning('Be aware that mu-metal shields are saturated by too large magnetic fields and will not work afterwards. '
                    'Are you sure you want to go to more than 1.5 T?', ResourceWarning, 'cqed/cqed/custom_pysweep_functions/magnet', 20)
    
    if np.sqrt(Bx**2 + By**2 + Bz**2) > max_field_strength:
        return bool(False)
    
    else:
        return bool(True)


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

@MakeMeasurementFunction([DataParameter(name='r', unit='T'), 
                          DataParameter(name='theta', unit='deg'),
                          DataParameter(name='phi', unit='deg')])
def measure_magnet_components_sph(d):
    ''' Measure the x, y, z component of the magnet, convert to spherical using the 
    ISO 80000-2:2009 physics convention for the (r, theta, phi) <--> (x, y, z) definition 
    and return as list. 
    output: [r, theta, phi] 
    r (float): magentic field strength, unit: T
    theta (float): inclination angle, unit: degrees, 0 <= theta <= 180
    phi (float): the azimuth (in plane) angle, unit: degrees, range: 0 <= phi < 360
    '''
    station = d['STATION']
    x = station.mgnt.x_measured()+1e-9 #avoiding dividing by true zero 
    y = station.mgnt.y_measured()+1e-9
    z = station.mgnt.z_measured()+1e-9

    r_meas = np.sqrt(x**2+y**2+z**2)
    phi_meas = np.arctan2(y,x)/np.pi*180
    if phi_meas < 0:
        phi_meas += 360
    theta_meas = np.arccos(z/r_meas)/np.pi*180

    return [r_meas, theta_meas, phi_meas] 

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

    if max_field_strength > 1.5:
        showwarning('Be aware that mu-metal shields are saturated by too large magnetic fields and will not work afterwards.'
                    'Are you sure you want to go to more than 1.5 T?', ResourceWarning, 'cqed/cqed/custom_pysweep_functions/magnet', 60)

    @MakeMeasurementFunction([])
    def point_function(d):
        return points, []
    
    @MakeMeasurementFunction([])
    def set_function(phi, d):

        assert max_field_strength>r>0., 'The field amplitude must not exceed {} and be larger than 0.' \
                                        ' Upper limit can be adjusted with kwarg: max_field_strength.' \
                                        ' Proceed with caution (Mu-metal shields do not appreciate high fields!)'.format(max_field_strength)
        assert 0.<=theta<=180., 'The inclination angle must be equal or lager than 0 and smaller or equal than 180. Change setting!'  
        assert 0.<=phi<=2*180., 'The azimuth angle must be equal or larger than 0 and smaller or equal than 360. Change setting!'  
        
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


def sweep_theta(r, phi, points, max_field_strength=1.5, offset_x=0, offset_y=0, offset_z=0):
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

    if max_field_strength > 1.5:
        showwarning('Be aware that mu-metal shields are saturated by too large magnetic fields and will not work afterwards.'
                    'Are you sure you want to go to more than 1.5 T?', ResourceWarning, 'cqed/cqed/custom_pysweep_functions/magnet', 109)

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

        x = r*np.sin(np.radians(theta))*np.cos(np.radians(phi)) + offset_x
        y = r*np.sin(np.radians(theta))*np.sin(np.radians(phi)) + offset_y
        z = r*np.cos(np.radians(theta)) + offset_z         
        
        station.mgnt.x_target(x)
        station.mgnt.y_target(y)
        station.mgnt.z_target(z)

        station.mgnt.ramp(mode='safe')
        
        return []
    
    return SweepObject(set_function = set_function, unit = 'degrees', label = 'theta', point_function = point_function, dataparameter=None )


def sweep_theta_with_alignment(r, phi, points, aligner, optimization_at=None, optimize_first=True, max_field_strength=1.5, offset_x=0, offset_y=0, offset_z=0):
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

    if max_field_strength > 1.5:
        showwarning('Be aware that mu-metal shields are saturated by too large magnetic fields and will not work afterwards.'
                    'Are you sure you want to go to more than 1.5 T?', ResourceWarning, 'cqed/cqed/custom_pysweep_functions/magnet', 109)
    if optimize_first:
        aligner.optimize_x()
        #ToDo: adjustment of theta and phi after field optimization

    if optimization_at is None:
        # default optimization of angle every 5 steps
        optimization_at = np.arange(points[0], points[-1], 5)
    optimization_value_index = 0

    @MakeMeasurementFunction([])
    def point_function(d):
        return points, []
    
    @MakeMeasurementFunction([])
    def set_function(theta, d, optimization_value_index=optimization_value_index, optimization_at=optimization_at):
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

        if station.mgnt.x_measured() !=0:
            offset_x = station.mgnt.x_measured() - r*np.sin(np.radians(theta))*np.cos(np.radians(phi))

        x = r*np.sin(np.radians(theta))*np.cos(np.radians(phi)) + offset_x
        y = r*np.sin(np.radians(theta))*np.sin(np.radians(phi)) + offset_y
        z = r*np.cos(np.radians(theta)) + offset_z         
        
        station.mgnt.x_target(x)
        station.mgnt.y_target(y)
        station.mgnt.z_target(z)

        station.mgnt.ramp(mode='safe')

        start = station.vna.S21.start()
        stop  = station.vna.S21.stop()
        npts = station.vna.S21.npts()
        power = station.vna.S21.power()
        bw = station.vna.S21.bandwidth()
        avg = station.vna.S21.avg()

        if theta >= optimization_at[optimization_value_index]:
            aligner.optimize_x()
            optimization_value_index += 1
            #TODO: update theta and phi after field optimization

        station.vna.S21.start(start)
        station.vna.S21.stop(stop)
        station.vna.S21.npts(npts)
        station.vna.S21.power(power)
        station.vna.S21.bandwidth(bw)
        station.vna.S21.avg(avg)
        
        return []
    
    return SweepObject(set_function = set_function, unit = 'degrees', label = 'theta', point_function = point_function, dataparameter=None )


def sweep_r(phi, theta, points, max_field_strength=1.5, offset_x=0, offset_y=0, offset_z=0):
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

    if max_field_strength > 1.5:
        showwarning('Be aware that mu-metal shields are saturated by too large magnetic fields and will not work afterwards.'
                    'Are you sure you want to go to more than 1.5 T?', ResourceWarning, 'cqed/cqed/custom_pysweep_functions/magnet', 162)

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
        
        x = r*np.sin(np.radians(theta))*np.cos(np.radians(phi)) + offset_x
        y = r*np.sin(np.radians(theta))*np.sin(np.radians(phi)) + offset_y
        z = r*np.cos(np.radians(theta)) + offset_z        
        
        station.mgnt.x_target(x)
        station.mgnt.y_target(y)
        station.mgnt.z_target(z)

        station.mgnt.ramp(mode='safe')
        
        return []
    
    return SweepObject(set_function = set_function, unit = "T", label = "r", point_function = point_function, dataparameter=None )

#TODO:
# Generate function, which ramps the field, optimizes the angle at predefined
# values and also saves the optimization data in the QCoDeS database

def sweep_r_with_alignment(theta, phi, points, FieldAligner, optimization_at=None, optimize_first=True, max_field_strength=1.5):

    if max_field_strength > 1.5:
        showwarning(
            'Be aware that mu-metal shields are saturated by too large magnetic fields and will not work afterwards.'
            'Are you sure you want to go to more than 1.5 T?', ResourceWarning,
            'cqed/cqed/custom_pysweep_functions/magnet', 225)

    if optimize_first:
        FieldAligner.optimize_x()
        #ToDo: adjustment of theta and phi after field optimization

    if optimization_at is None:
        # default optimization of angle every 100 mT
        optimization_at = np.arange(points[0], points[-1], 100e-3)
    optimization_value_index = 1

    @MakeMeasurementFunction([])
    def point_function(d):
        return points, []

    @MakeMeasurementFunction([])
    def set_function(r, d):
        # Here we use the ISO 80000-2:2009 physics convention for the (r, theta, phi) <--> (x, y, z) definition.
        # Note that r is the radial distance, theta the inclination and phi the azimuth (in-plane) angle.
        # For uniqueness, we restrict the parameter choice to r>=0, 0<= theta <= pi and 0<= phi <= 2pi.
        # The units are: r in T, theta in degrees, phi in degrees.

        assert max_field_strength > r > 0., 'The field amplitude must not exceed {} and be lager than 0.' \
                                            ' Upper limit can be adjusted with kwarg: max_field_strength.' \
                                            ' Proceed with caution (Mu-metal shields do not appreciate high fields!)'.format(
            max_field_strength)
        assert 0. <= theta <= 180., 'The inclination angle must be equal or lager than 0 and smaller or equal than 180. Change setting!'
        assert 0. <= phi <= 2 * 180., 'The azimuth angle must be equal or lager than 0 and smaller or equal than 360. Change setting!'

        station = d['STATION']

        x = r * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
        y = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
        z = r * np.cos(np.radians(theta))

        station.mgnt.x_target(x)
        station.mgnt.y_target(y)
        station.mgnt.z_target(z)

        station.mgnt.ramp(mode='safe')

        if r >= optimization_at[optimization_value_index]:
            FieldAligner.optimize_x()
            optimization_value_index += 1
            #TODO: update theta and phi after field optimization
        return []

    return SweepObject(set_function=set_function, unit="T", label="r", point_function=point_function,
                       dataparameter=None)
