import numpy as np
import broadbean as bb

from pytopo.qctools.hard_sweep import HardSweep

ramp = bb.PulseAtoms.ramp

class BluePrints(object):

    FILL_TOKENS = ['FILL.1', 'FILL.2', 'FILL.3', 'FILL.4']
    
    def __init__(self, chan_map, sample_rate=1e9, autofill=True, length=None):
        self.bps = {}
        self.map = {}
        
        for i, lst in chan_map.items():
            self.bps[i] = bb.BluePrint()
            self.bps[i].setSR(sample_rate)
            for j, name in enumerate(lst):                    
                if name is not None:
                    self.map[name] = (i, j)

                if autofill and length is not None and name in self.FILL_TOKENS:
                    self[name].insertSegment(0, ramp, (0, 0), dur=length, name=name+'_segment')
                    
    def __getitem__(self, name):
        if self.map[name][1] == 0:
            return self.bps[self.map[name][0]]
        else:
            return getattr(self.bps[self.map[name][0]], 'marker{}'.format(self.map[name][1]))
        
    def __setitem__(self, name, value):
        if self.map[name][1] == 0:
            self.bps[self.map[name][0]] = value
        else:
            setattr(self.bps[self.map[name][0]], 'marker{}'.format(self.map[name][1]), value)
        
    def __call__(self):
        return list(self.bps.items())


class BroadBeanSequence(HardSweep):
    
    chan_map = {}
    sweep_dims = []

    chan_settings = {
        1 : {
            'Vpp' : 1, 
            'offset' : 0,
        }, 
        2 : {
            'Vpp' : 1, 
            'offset' : 0,
        },
        3 : {
            'Vpp' : 1, 
            'offset' : 0,
        },
        4 : {
            'Vpp' : 1, 
            'offset' : 0,
        },
    }

    sweep_wait = 'first'
    sweep_repeat = 'sequence'
        
    def __init__(self, name, awg, chan_map=None, **kw):
        self.awg = awg
        
        if chan_map is not None:
            self.chan_map = chan_map
        
        kw['sweep_dims'] = self.__class__.sweep_dims
        super().__init__(name, **kw)
        
        self.add_parameter('sample_rate', get_cmd=None, set_cmd=None, 
                           unit='GS/s', initial_value=1e9)
        
    def sequence(self):
        raise NotImplementedError
        
        
    def setup(self, program_awg=True, start_awg=True):
        super().setup()
        
        self.awg.stop()
        
        if program_awg:
            self.awg.clock_freq(self.sample_rate())
            seq = self.sequence()
            
            for ch_no, ch_set in self.chan_settings.items():
                seq.setChannelAmplitude(ch_no, ch_set['Vpp'])
                seq.setChannelOffset(ch_no, ch_set['offset'])
            seq.setSR(self.sample_rate())

            if self.sweep_wait == 'first':
                seq.setSequencingTriggerWait(1, 1)
            elif self.sweep_wait == 'off':
                seq.setSequencingTriggerWait(1, 0)
            elif self.sweep_wait == None:
                pass
            else:
                raise ValueError("Unknown sweep_wait setting '{}".format(self.sweep_wait))

            if self.sweep_repeat == 'sequence':
                seq.setSequencingGoto(seq.length_sequenceelements, 1)
            elif self.sweep_repeat == None:
                pass
            else:
                raise ValueError("Unknown sweep_repeat setting '{}".format(self.sweep_repeat))


            pkg = seq.outputForAWGFile()
            self.awg.make_send_and_load_awg_file(*pkg[:])
        
            for ch_no, ch_desc in self.chan_map.items():
                self.awg.set('ch{}_state'.format(ch_no), 1)

            for ch_no, ch_set in self.chan_settings.items():
                self.awg.set('ch{}_amp'.format(ch_no), ch_set['Vpp'])
                self.awg.set('ch{}_offset'.format(ch_no), ch_set['offset'])

        if start_awg:
            self.awg.start()

