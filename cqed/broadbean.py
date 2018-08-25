import numpy as np
import broadbean as bb

from qcodes.instrument_drivers.tektronix.AWG5208 import AWG5208
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014 as AWG5014

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
            'marker_hi' : [1, 1, 1, 1],
            'marker_lo' : [0, 0, 0, 0],
        }, 
        2 : {
            'Vpp' : 1, 
            'offset' : 0,
            'marker_hi' : [1, 1, 1, 1],
            'marker_lo' : [0, 0, 0, 0],
        },
        3 : {
            'Vpp' : 1, 
            'offset' : 0,
            'marker_hi' : [1, 1, 1, 1],
            'marker_lo' : [0, 0, 0, 0],
        },
        4 : {
            'Vpp' : 1, 
            'offset' : 0,
            'marker_hi' : [1, 1, 1, 1],
            'marker_lo' : [0, 0, 0, 0],
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
        
        
    def setup(self, program_awg=True, start_awg=True, stop_awg=True):
        super().setup()
        
        if stop_awg:
            self.awg.stop()
        
        if program_awg:      
            
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

            if isinstance(self.awg, AWG5014):
                pkg = seq.outputForAWGFile()
                self.awg.make_send_and_load_awg_file(*pkg[:])

                for ch_no in self.chan_map.items():
                    self.awg.set('ch{}_state'.format(ch_no), 1)

                for ch_no, ch_set in self.chan_settings.items():
                    self.awg.set('ch{}_amp'.format(ch_no), ch_set['Vpp'])
                    self.awg.set('ch{}_offset'.format(ch_no), ch_set['offset'])

                self.awg.clock_freq(self.sample_rate())

            elif isinstance(self.awg, AWG5208):
                # forge the sequence
                forged_sequence = seq.forge()

                # create a sequence file
                seqx_file = self.awg.make_SEQX_from_forged_sequence(
                    forged_sequence, [1 for i in self.chan_map.keys()], seq.name)
                seqx_file_name = f'{seq.name}.seqx'

                # clear lists of sequences and waveforms on the instrument in order
                # to prevent cluttering
                self.awg.clearSequenceList()
                self.awg.clearWaveformList()

                # send the sequence file to the instrument and load it
                self.awg.sendSEQXFile(seqx_file, filename=seqx_file_name)
                self.awg.loadSEQXFile(seqx_file_name)

                self.awg.sample_rate(self.sample_rate())

                # load seqs to channels
                for ch_no, ch_desc in self.chan_map.items():
                    chan = self.awg.channels[ch_no-1]

                    track_number = 1
                    chan.setSequenceTrack(seq.name, track_number)

                    # NOTE: assuming the sequence file contains a single waveform
                    waveform_list = self.awg.waveformList
                    if len(waveform_list) != 1:
                        raise Exception('There are 0 or more than 1 waveforms in the list. Only single waveform assignment is supported.')
                    waveform_name = waveform_list[0]
                    chan.setWaveform(waveform_name)  # usually wfm_1_1_1

                    chan.resolution(12)
                    chan.set('state', 1)

                for ch_no, ch_set in self.chan_settings.items():
                    self.awg.channels[ch_no-1].set('awg_amplitude', ch_set['Vpp'])
                    for i in range(1, 5):
                        self.awg.channels[ch_no-1].set('marker{}_high'.format(i), ch_set['marker_hi'][i-1])
                        self.awg.channels[ch_no-1].set('marker{}_low'.format(i), ch_set['marker_lo'][i-1])

            
        if start_awg:
            if isinstance(self.awg, AWG5014):
                self.awg.start()
            elif isinstance(self.awg, AWG5208):
                self.awg.play()

