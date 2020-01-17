def setup_frq_sweep(fstart, fstop, fpts, chan='S21', bw=None, navgs=None, pwr=None):
    """
    Setup a VNA trace.
    
    assumes that a channel with name chan is already created.
    """
    vna = qc.Station.default.vna
    trace = getattr(vna.channels, chan)
    
    fvals = np.linspace(fstart, fstop, fpts)
    trace.start(fstart)
    trace.stop(fstop)
    trace.npts(fpts)
    if navgs is not None:
        trace.avg(navgs)
    if bw is not None:
        trace.bandwidth(bw)
    if pwr is not None:
        trace.power(pwr)
    trace.autoscale()
    
    if not vna.rf_power():
        vna.rf_on()
    
    return fvals

def take_trace(chan='S21', plot=False):
    """
    Get the data of a currently measured trace.
    The trace has to be setup already.
    
    Returns magnitude (in dB) and phase (in rad).
    
    If plot is true, make a simple plot of the magnitude vs frequency.
    """
    vna = qc.Station.default.vna
    trace = getattr(vna.channels, chan)
    
    fvals = np.linspace(trace.start(), trace.stop(), trace.npts())
    mag, phase = trace.trace_mag_phase()
    
    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(fvals*1e-9, 20*np.log10(mag))
        ax.grid(dashes=[1,1])
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dBm)')
        
    return mag, phase

    def find_resonance(plot=False):
    global current_objective
    '''Crude estimate of the resonance frequency and its width. Could use a fancier fit, but would be less robust.
    Inputs:
    plot (boolean): plots the VNA linescan
    '''
    fval = setup_frq_sweep(fstart=current_objective-df, fstop=current_objective+df, fpts=501, pwr=-20, bw=1e2, navgs=1)
    mag, phase = take_trace(plot=plot)
    vna.rf_off()
    f0 = fval[np.argmin(mag)] #crude resonance frequency estimate
    return f0

def find_resonance_advanced(current_objective, df, fpts=501, pwr=-20, bw=1e2, navgs=1, plot=False,chan = "S21"):
    fstart=current_objective-df
    fstop=current_objective+df
    station.vna.timeout(fpts/bw*1.1)
    fval = setup_frq_sweep(fstart, fstop, fpts, chan, bw, navgs, pwr)
    mag, phase = take_trace(plot=plot)
    vna.rf_off()
    f0 = fval[np.argmin(mag)] 
    return f0

    def optimize_x(max_magnitude, wiggle_step, min_wiggle, observer_fn, plot=True, return_extra=True):
    global current_objective
    ''' Wiggle the x-axis of the magnetic field up and down until the objective is optimized to the desired accuracy.
    Picture a Roombah, I have been told.
    Inputs: 
    max_magnitude (float): maximal magnetic field value for x, unit: T
    wiggle_step (float): initial step size for changes in magnetic field, unit: T
    min_wiggle (float): smallest field step taken after which the protocol terminates, unit: T
    min_wiggle (float): smallest field step taken after which the protocol terminates, unit: T
    observer_fn: a callable which gets called after each small step and measures the objective
    '''
    objectives_meas = []
    locations = []
    
    nsteps = 1
    direction = 1
    
    current_objective = observer_fn()
    
    objectives_meas.append(current_objective)
    locations.append(station.mgnt.x_measured())
    
    while wiggle_step>min_wiggle:
        pos = station.mgnt.x_measured()
        if abs(pos+direction*wiggle_step) > max_magnitude:
            raise Exception('X-axis magnitude limit reached.')
            
        print(current_objective*1e-9, '{0:.6f}'.format(station.mgnt.x_measured()), '->', '{0:.6f}'.format(pos+direction*wiggle_step))
        station.mgnt.x_target(pos+direction*wiggle_step)
        station.mgnt.ramp(mode='safe')
        time.sleep(1.) 
        newpos = station.mgnt.x_measured()
        if np.abs(pos-newpos) < 0.8*wiggle_step:
            continue
        
        new_objective = observer_fn()
        objectives_meas.append(new_objective)
        locations.append(station.mgnt.x_measured())
        nsteps+=1
        if new_objective < current_objective:
            current_objective = new_objective
            direction = -1*direction
            if direction == 1:
                wiggle_step=wiggle_step/2
        else:
            current_objective = new_objective
        
            
    optimum = current_objective        
    print('nstep', nsteps)
    extra = {'objectives': objectives_meas, 
            'fields': locations}
    
    if plot:
        plt_xs = extra['fields']
        plt_ys = extra['objectives']
        plt.figure()
        plt.plot(
            plt_xs,
            plt_ys,
        )
        plt.scatter(
            plt_xs,
            plt_ys,
            c=extra['objectives']
        )
        plt.colorbar()

    if return_extra:
        return optimum, extra
    else:
        return optimum