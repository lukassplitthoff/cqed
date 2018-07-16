import logging

import numpy as np
import broadbean as bb
from .broadbean import BroadBeanSequence, BluePrints

logger = logging.getLogger(__name__)

ramp = bb.PulseAtoms.ramp
sine = bb.PulseAtoms.sine

class TriggeredReadoutSequence(BroadBeanSequence):
    
    sweep_dims = ['awg_sweep_idx', ]
    sweep_wait = 'off'
    
    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        
        self.add_parameter('pre_trigger_delay', set_cmd=None,
                           unit='s', initial_value=1e-6)
        self.add_parameter('trigger_len', set_cmd=None,
                           unit='s', initial_value=1e-7)
        self.add_parameter('post_trigger_delay', set_cmd=None, 
                           unit='s', initial_value=1e-6)
        self.add_parameter('seq_len', set_cmd=None, 
                           unit='s', initial_value=1e-5)
        self.add_parameter('readout_gate_len', set_cmd=None,
                           unit='s', initial_value=0)
        
    def setup(self, **kw):
        self.awg_sweep_idx([1,])
        super().setup(**kw)
        
    def sequence(self):
        elem = bb.Element()

        bps = BluePrints(self.chan_map, sample_rate=self.sample_rate(), length=self.seq_len())
        bps['ro_trigger'] = [(self.pre_trigger_delay(), self.trigger_len())]
        bps['ro_gate'] = [(self.pre_trigger_delay() + self.post_trigger_delay(), 
                           self.readout_gate_len())]

        try:
            bps['src_gate'] = [(self.pre_trigger_delay() + self.post_trigger_delay(), 
                                self.readout_gate_len())]
        except KeyError:
            logger.warning('No src_gate defined in channel map. Only readout gate pulse will be generated.')

        for n, bp in bps():
            elem.addBluePrint(n, bp)
            
        seq = bb.Sequence()
        seq.name = 'trig'
        seq.addElement(1, elem)

        seq.setSequencingTriggerWait(1, 0)
        seq.setSequencingGoto(1, 1)
        
        return seq


class TwoToneSSBSequence(BroadBeanSequence):
    
    sweep_dims = ['ssb_frequency', ]
    
    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        
        self.add_parameter('pre_delay', set_cmd=None,
                           unit='s', initial_value=1e-6)
        self.add_parameter('post_delay', set_cmd=None,
                           unit='s', initial_value=1e-6)
        self.add_parameter('trigger_len', set_cmd=None,
                           unit='s', initial_value=1e-7)
        self.add_parameter('ssb_start', set_cmd=None,
                           unit='Hz', initial_value=10e6)
        self.add_parameter('ssb_stop', set_cmd=None,
                           unit='Hz', initial_value=20e6)
        self.add_parameter('ssb_steps', set_cmd=None,
                           unit='', initial_value=11)
        self.add_parameter('ssb_amp', set_cmd=None,
                           unit='V', initial_value=0.5)
        self.add_parameter('seq_len', set_cmd=None, 
                           unit='s', initial_value=1e-5)
        
    def setup(self, **kw):
        self.ssb_frequency(np.linspace(self.ssb_start(), self.ssb_stop(), self.ssb_steps()))
        super().setup(**kw)
        
    def sequence(self):
        seq = bb.Sequence()
        seq.name = 'SSB'
        
        net_time = self.seq_len() - self.pre_delay() - self.post_delay()
        
        for i, f in enumerate(self.ssb_frequency()):
            elem = bb.Element()
            bps = BluePrints(self.chan_map, sample_rate=self.sample_rate())
            
            bps['src_I'].insertSegment(0, ramp, (0, 0), dur=self.pre_delay())
            bps['src_I'].insertSegment(1, sine, (f, self.ssb_amp(), 0, 0), dur=net_time)
            bps['src_I'].insertSegment(2, ramp, (0, 0), dur=self.post_delay())
            
            bps['src_Q'].insertSegment(0, ramp, (0, 0), dur=self.pre_delay())
            bps['src_Q'].insertSegment(1, sine, (f, self.ssb_amp(), 0, -np.pi/2), dur=net_time)
            bps['src_Q'].insertSegment(2, ramp, (0, 0), dur=self.post_delay())
            
            bps['ro_trigger'] = [(self.pre_delay(), self.trigger_len())]
            bps['ro_gate'] = [(self.pre_delay(), net_time)]
            bps['src_gate'] = [(self.pre_delay(), net_time)]
            
            for n, bp in bps():
                elem.addBluePrint(n, bp)
            
            seq.addElement(i+1, elem)
            
        return seq
