import broadbean as bb
from pytopo.awg_sequencing import broadbean as bbtools
from pytopo.awg_sequencing.broadbean import BluePrints, BroadBeanSequence
import numpy as np

ramp = bb.PulseAtoms.ramp
gaussian = bb.PulseAtoms.gaussian

class RabiSequence(BroadBeanSequence):
    """
    A sequence that consists of a single rectangular pulse followed by a readout pulse.

    required channels:
        'I' : analog output (qubit drive pulse)
        'ats_trigger' : marker for the alazar
        'ro_pulse' : readout marker 
        'qb_pulse' : qubit marker
    """
    name = 'rabi_sequence'

    def sequence(self, pulse_times=None, amplitudes=None, pulse_time=20e-9, amplitude=0.5, readout_time=2.6e-9, cycle_time = 10e-6,
                 pre_pulse_time=1e-6, after_pulse_time=0.02e-6,
                 alazar_trigger_time=100e-9, marker_buffer=20e-9, cavity_lifetime=0.3e-6):
        
        ii = 0
        if amplitudes:
            ii += 1
        if pulse_times:
            ii += -1

        if ii == 0:
            raise ValueError('pulse_times xor amplitudes has to be given.')
        elif ii == 1:
            pulse_times = np.ones_like(amplitudes)*pulse_time
        elif ii == -1:
            amplitudes = np.ones_like(pulse_times)*amplitude

        elements = []
        for amplitude, pulse_time in zip(amplitudes, pulse_times):
            pulse_time = np.round(pulse_time/1e-9)*1e-9
            t_pulse = pre_pulse_time + pulse_time + after_pulse_time
            
            bps = bbtools.BluePrints(chan_map=self.chan_map, length=cycle_time, sample_rate=self.SR)
            
            bps['I'].insertSegment(0, ramp, (0, 0), dur=pre_pulse_time)
            bps['I'].insertSegment(1, ramp, (amplitude, amplitude), dur=pulse_time)
            bps['I'].insertSegment(2, ramp, (0, 0), dur=cycle_time - pre_pulse_time - pulse_time)
            
            
            bps['qb_pulse'] = [(pre_pulse_time - marker_buffer, pulse_time + 2 * marker_buffer)]
            bps['ats_trigger'] = [(t_pulse+cavity_lifetime, alazar_trigger_time)]
            bps['ro_pulse'] = [(t_pulse, readout_time)]

            elements.append(bbtools.blueprints2element(bps))
        
        return bbtools.elements2sequence(elements, self.name)

class RamseySequence(BroadBeanSequence):
    """
    A sequence that consists of a single gaussian pulse followed by a readout pulse.

    required channels:
        'I' : analog output (qubit drive pulse)
        'ats_trigger' : marker for the alazar
        'ro_pulse' : readout marker 
        'qb_pulse' : qubit marker
    """
    name = 'ramsey_sequence'

    def sequence(self, delays, pulse_time, readout_time, amplitude=0.5, cycle_time = 10e-6,
                 pre_pulse_time=1e-6, after_pulse_time=0.02e-6,
                 alazar_trigger_time=100e-9, marker_buffer=20e-9, cavity_lifetime=0.3e-6):
        
         
        pulse_time = np.round(pulse_time/1e-9)*1e-9
        elements = []
        for D in delays:
            D = np.round(D/1e-9)*1e-9
            t_pulse = pre_pulse_time + pulse_time + D + pulse_time + after_pulse_time      
            bps = bbtools.BluePrints(chan_map=self.chan_map, length=cycle_time, sample_rate=self.SR)
            
            bps['I'].insertSegment(0, ramp, (0, 0), dur=pre_pulse_time)
            bps['I'].insertSegment(1, ramp, (amplitude,amplitude), dur=pulse_time)
            bps['I'].insertSegment(2, ramp, (0, 0), dur=D)
            bps['I'].insertSegment(3, ramp, (amplitude,amplitude), dur=pulse_time)
            bps['I'].insertSegment(4, ramp, (0, 0), dur=cycle_time - (pre_pulse_time+2*pulse_time+D))
            
            bps['qb_pulse'] = [(pre_pulse_time - marker_buffer, pulse_time + 2 * marker_buffer), 
                               (pre_pulse_time + pulse_time + D - marker_buffer, pulse_time + 2 * marker_buffer)]
            bps['ats_trigger'] = [(t_pulse+cavity_lifetime, alazar_trigger_time)]
            bps['ro_pulse'] = [(t_pulse, readout_time)]

            elements.append(bbtools.blueprints2element(bps))
        
        return bbtools.elements2sequence(elements, self.name)
    

class T1Sequence(BroadBeanSequence):
    """
    A sequence that...

    required channels:
        'I' : analog output (qubit drive pulse)
        'ats_trigger' : marker for the alazar
        'ro_pulse' : readout marker 
        'qb_pulse' : qubit marker
    """
    name = 'T1_sequence'

    def sequence(self, delays, pulse_time, readout_time, amplitude=0.5, cycle_time = 10e-6,
                 pre_pulse_time=1e-6, after_pulse_time=0.02e-6,
                 alazar_trigger_time=100e-9, marker_buffer=20e-9, cavity_lifetime=0.3e-6):
        
        pulse_time = np.round(pulse_time/1e-9)*1e-9
        t_pulse = pre_pulse_time + pulse_time + after_pulse_time
                
        elements = []
        for D in delays:
            D = np.round(D/1e-9)*1e-9
            bps = bbtools.BluePrints(chan_map=self.chan_map, length=cycle_time, sample_rate=self.SR)
            bps['I'].insertSegment(0, ramp, (0, 0), dur=pre_pulse_time)
            bps['I'].insertSegment(1, ramp, (amplitude,amplitude), dur=pulse_time)
            bps['I'].insertSegment(2, ramp, (0, 0), dur=cycle_time - pre_pulse_time - pulse_time)
            
            bps['qb_pulse'] = [(pre_pulse_time - marker_buffer, pulse_time + 2 * marker_buffer)]
            bps['ats_trigger'] = [(t_pulse+D+cavity_lifetime, alazar_trigger_time)]
            bps['ro_pulse'] = [(t_pulse+D, readout_time)]

            elements.append(bbtools.blueprints2element(bps))
        
        return bbtools.elements2sequence(elements, self.name)


class EchoSequence(BroadBeanSequence):
    """
    A sequence that consists of a ...

    required channels:
        'I' : analog output (qubit drive pulse)
        'ats_trigger' : marker for the alazar
        'ro_pulse' : readout marker 
        'qb_pulse' : qubit marker
    """
    name = 'echo_sequence'

    def sequence(self, delays, pulse_time, readout_time, amplitude=0.5, cycle_time = 10e-6,
                 pre_pulse_time=1e-6, after_pulse_time=0.02e-6,
                 alazar_trigger_time=100e-9, marker_buffer=20e-9, cavity_lifetime=0.3e-6):
        
         
        pulse_time = np.round(pulse_time/1e-9)*1e-9
        elements = []
        for D in delays:
            D = np.round(D/1e-9)*1e-9                      
            t_pulse = pre_pulse_time + 4*pulse_time + D + after_pulse_time      
            bps = bbtools.BluePrints(chan_map=self.chan_map, length=cycle_time, sample_rate=self.SR)
            
            bps['I'].insertSegment(0, ramp, (0, 0), dur=pre_pulse_time)
            bps['I'].insertSegment(1, ramp, (amplitude,amplitude), dur=pulse_time)
            bps['I'].insertSegment(2, ramp, (0, 0), dur=D/2)
            bps['I'].insertSegment(3, ramp, (amplitude,amplitude), dur=pulse_time*2)
            bps['I'].insertSegment(2, ramp, (0, 0), dur=D/2)
            bps['I'].insertSegment(1, ramp, (amplitude,amplitude), dur=pulse_time)
            bps['I'].insertSegment(4, ramp, (0, 0), dur=cycle_time - (pre_pulse_time+4*pulse_time+D))
            
            bps['qb_pulse'] = [(pre_pulse_time - marker_buffer, pulse_time + 2 * marker_buffer), 
                               (pre_pulse_time + pulse_time + D/2 - marker_buffer, 2*pulse_time + 2 * marker_buffer), 
                              (pre_pulse_time + 3*pulse_time + D - marker_buffer, pulse_time + 2 * marker_buffer)]
            bps['ats_trigger'] = [(t_pulse+cavity_lifetime, alazar_trigger_time)]
            bps['ro_pulse'] = [(t_pulse, readout_time)]

            elements.append(bbtools.blueprints2element(bps))
        
        return bbtools.elements2sequence(elements, self.name)