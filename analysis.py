#!/usr/bin/env python

from . import Amp
from .utilities import now, hash_images, make_filename
import os
import numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def plot_sensitivity(load,
                     images,
                     d=0.0001,
                     label='sensitivity',
                     dblabel=None,
                     plotfile=None,
                     overwrite=False,
                     energy_coefficient=1.0,
                     force_coefficient=0.04):
    """Returns the plot of loss function in terms of perturbed parameters.

    Takes the load file and images. Any other keyword taken by the Amp
    calculator can be fed to this class also.

    Parameters
    ----------
    load : str
        Path for loading an existing ".amp" file. Should be fed like
        'load="filename.amp"'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    d : float
        The amount of perturbation in each parameter.
    label : str
        Default prefix/location used for all files.
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    plotfile : Object
        File for the plot.
    overwrite : bool
        If a plot or an script containing values found overwrite it.
    energy_coefficient : float
        Coefficient of energy loss in the total loss function.
    force_coefficient : float
        Coefficient of force loss in the total loss function.
    """

    from amp.model import LossFunction

    calc = Amp.load(file=load)

    if plotfile is None:
        plotfile = make_filename(label, '-plot.pdf')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.' % plotfile)

    calc.dblabel = label if dblabel is None else dblabel

    if force_coefficient == 0.:
        calculate_derivatives = False
    else:
        calculate_derivatives = True

    calc._log('\nAmp sensitivity analysis started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
            images=images,
            parallel=calc._parallel,
            log=calc._log,
            calculate_derivatives=calculate_derivatives)

    vector = calc.model.vector.copy()

    lossfunction = LossFunction(energy_coefficient=energy_coefficient,
                                force_coefficient=force_coefficient,
                                parallel=calc._parallel,
                                )
    calc.model.lossfunction = lossfunction

    # Set up local loss function.
    calc.model.lossfunction.attach_model(
                calc.model,
                fingerprints=calc.descriptor.fingerprints,
                fingerprintprimes=calc.descriptor.fingerprintprimes,
                images=images)

    originalloss = calc.model.lossfunction.get_loss(
        vector, lossprime=False)['loss']

    calc._log('\n Perturbing parameters...', tic='perturb')

    allparameters = []
    alllosses = []
    num_parameters = len(vector)

    for count in range(num_parameters):
        calc._log('parameter %i out of %i' % (count + 1, num_parameters))
        parameters = []
        losses = []
        # parameter is perturbed -d and loss function calculated.
        vector[count] -= d
        parameters.append(vector[count])
        perturbedloss = calc.model.lossfunction.get_loss(
            vector, lossprime=False)['loss']
        losses.append(perturbedloss)

        vector[count] += d
        parameters.append(vector[count])
        losses.append(originalloss)
        # parameter is perturbed +d and loss function calculated.
        vector[count] += d
        parameters.append(vector[count])
        perturbedloss = calc.model.lossfunction.get_loss(
            vector, lossprime=False)['loss']
        losses.append(perturbedloss)

        allparameters.append(parameters)
        alllosses.append(losses)
        # returning back to the original value.
        vector[count] -= d

    calc._log('...parameters perturbed and loss functions calculated',
              toc='perturb')

    calc._log('Plotting loss function vs perturbed parameters...',
              tic='plot')

    with PdfPages(plotfile) as pdf:
        count = 0
        for parameter in vector:
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            ax.plot(allparameters[count],
                    alllosses[count],
                    marker='o', linestyle='--', color='b',)

            xmin = allparameters[count][0] - \
                0.1 * (allparameters[count][-1] - allparameters[count][0])
            xmax = allparameters[count][-1] + \
                0.1 * (allparameters[count][-1] - allparameters[count][0])
            ymin = min(alllosses[count]) - \
                0.1 * (max(alllosses[count]) - min(alllosses[count]))
            ymax = max(alllosses[count]) + \
                0.1 * (max(alllosses[count]) - min(alllosses[count]))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

            ax.set_xlabel('parameter no %i' % count)
            ax.set_ylabel('loss function')
            pdf.savefig(fig)
            pyplot.close(fig)
            count += 1

    calc._log(' ...loss functions plotted.', toc='plot')


def plot_parity(load,
                images,
                label='parity',
                dblabel=None,
                plot_forces=True,
                plotfile=None,
                color='b.',
                overwrite=False,
                returndata=False,
                energy_coefficient=1.0,
                force_coefficient=0.04):
    """Makes a parity plot of Amp energies and forces versus real energies and
    forces.

    Parameters
    ----------
    load : str
        Path for loading an existing ".amp" file. Should be fed like
        'load="filename.amp"'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    label : str
        Default prefix/location used for all files.
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    plot_forces : bool
        Determines whether or not forces should be plotted as well.
    plotfile : Object
        File for the plot.
    color : str
        Plot color.
    overwrite : bool
        If a plot or an script containing values found overwrite it.
    returndata : bool
        Whether to return a reference to the figures and their data or not.
    energy_coefficient : float
        Coefficient of energy loss in the total loss function.
    force_coefficient : float
        Coefficient of force loss in the total loss function.
    """

    calc = Amp.load(file=load, label=label, dblabel=dblabel)

    if plotfile is None:
        plotfile = make_filename(label, '-plot.pdf')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile)

    if (force_coefficient != 0.) or (plot_forces is True):
        calculate_derivatives = True
    else:
        calculate_derivatives = False

    calc._log('\nAmp parity plot started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images, log=calc._log)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
            images=images,
            parallel=calc._parallel,
            log=calc._log,
            calculate_derivatives=calculate_derivatives)

    calc._log('Calculating potential energies...', tic='pot-energy')
    energy_data = {}
    for hash, image in images.iteritems():
        amp_energy = calc.model.calculate_energy(
            calc.descriptor.fingerprints[hash])
        actual_energy = image.get_potential_energy(apply_constraint=False)
        energy_data[hash] = [actual_energy, amp_energy]
    calc._log('...potential energies calculated.', toc='pot-energy')

    min_act_energy = min([energy_data[hash][0]
                         for hash, image in images.iteritems()])
    max_act_energy = max([energy_data[hash][0]
                         for hash, image in images.iteritems()])

    if plot_forces is False:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    calc._log('Plotting energies...', tic='energy-plot')
    for hash, image in images.iteritems():
        ax.plot(energy_data[hash][0], energy_data[hash][1], color)
    # draw line
    ax.plot([min_act_energy, max_act_energy],
            [min_act_energy, max_act_energy],
            'r-',
            lw=0.3,)
    ax.set_xlabel("ab initio energy, eV")
    ax.set_ylabel("Amp energy, eV")
    ax.set_title("Energies")
    calc._log('...energies plotted.', toc='energy-plot')

    if plot_forces is True:
        ax = fig.add_subplot(212)

        calc._log('Calculating forces...', tic='forces')
        force_data = {}
        for hash, image in images.iteritems():
            amp_forces = \
                calc.model.calculate_forces(
                    calc.descriptor.fingerprints[hash],
                    calc.descriptor.fingerprintprimes[hash])
            actual_forces = image.get_forces(apply_constraint=False)
            force_data[hash] = [actual_forces, amp_forces]
        calc._log('...forces calculated.', toc='forces')

        min_act_force = min([force_data[hash][0][index][k]
                            for hash, image in images.iteritems()
                            for index in range(len(image))
                            for k in range(3)])

        max_act_force = max([force_data[hash][0][index][k]
                            for hash, image in images.iteritems()
                            for index in range(len(image))
                            for k in range(3)])

        calc._log('Plotting forces...', tic='force-plot')
        for hash, image in images.iteritems():
            for index in range(len(image)):
                for k in range(3):
                    ax.plot(force_data[hash][0][index][k],
                            force_data[hash][1][index][k], color)
        # draw line
        ax.plot([min_act_force, max_act_force],
                [min_act_force, max_act_force],
                'r-',
                lw=0.3,)
        ax.set_xlabel("ab initio force, eV/Ang")
        ax.set_ylabel("Amp force, eV/Ang")
        ax.set_title("Forces")
        calc._log('...forces plotted.', toc='force-plot')

    fig.savefig(plotfile)

    if returndata:
        if plot_forces is False:
            return fig, energy_data
        else:
            return fig, energy_data, force_data


def plot_error(load,
               images,
               label='error',
               dblabel=None,
               plot_forces=True,
               plotfile=None,
               color='b.',
               overwrite=False,
               returndata=False,
               energy_coefficient=1.0,
               force_coefficient=0.04):
    """Makes an error plot of Amp energies and forces versus real energies and
    forces.

    Parameters
    ----------
    load : str
        Path for loading an existing ".amp" file. Should be fed like
        'load="filename.amp"'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    label : str
        Default prefix/location used for all files.
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    plot_forces : bool
        Determines whether or not forces should be plotted as well.
    plotfile : Object
        File for the plot.
    color : str
        Plot color.
    overwrite : bool
        If a plot or an script containing values found overwrite it.
    returndata : bool
        Whether to return a reference to the figures and their data or not.
    energy_coefficient : float
        Coefficient of energy loss in the total loss function.
    force_coefficient : float
        Coefficient of force loss in the total loss function.
    """

    calc = Amp.load(file=load)

    if plotfile is None:
        plotfile = make_filename(label, '-plot.pdf')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile)

    calc.dblabel = label if dblabel is None else dblabel

    if (force_coefficient != 0.) or (plot_forces is True):
        calculate_derivatives = True
    else:
        calculate_derivatives = False

    calc._log('\nAmp error plot started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images, log=calc._log)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
            images=images,
            parallel=calc._parallel,
            log=calc._log,
            calculate_derivatives=calculate_derivatives)

    calc._log('Calculating potential energy errors...', tic='pot-energy')
    energy_data = {}
    for hash, image in images.iteritems():
        no_of_atoms = len(image)
        amp_energy = calc.model.calculate_energy(
            calc.descriptor.fingerprints[hash])
        actual_energy = image.get_potential_energy(apply_constraint=False)
        act_energy_per_atom = actual_energy / no_of_atoms
        energy_error = abs(amp_energy - actual_energy) / no_of_atoms
        energy_data[hash] = [act_energy_per_atom, energy_error]
    calc._log('...potential energy errors calculated.', toc='pot-energy')

    # calculating energy per atom rmse
    energy_square_error = 0.
    for hash, image in images.iteritems():
        energy_square_error += energy_data[hash][1] ** 2.
    energy_per_atom_rmse = np.sqrt(energy_square_error / len(images))

    min_act_energy_per_atom = min([energy_data[hash][0]
                                   for hash, image in images.iteritems()])
    max_act_energy_per_atom = max([energy_data[hash][0]
                                   for hash, image in images.iteritems()])

    if plot_forces is False:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    calc._log('Plotting energy errors...', tic='energy-plot')
    for hash, image in images.iteritems():
        ax.plot(energy_data[hash][0], energy_data[hash][1], color)
    # draw horizontal line for rmse
    ax.plot([min_act_energy_per_atom, max_act_energy_per_atom],
            [energy_per_atom_rmse, energy_per_atom_rmse],
            color='black', linestyle='dashed', lw=1,)
    ax.text(max_act_energy_per_atom,
            energy_per_atom_rmse,
            'energy rmse = %6.5f' % energy_per_atom_rmse,
            ha='right',
            va='bottom',
            color='black')
    ax.set_xlabel("ab initio energy (eV) per atom")
    ax.set_ylabel("$|$ab initio energy - Amp energy$|$ / number of atoms")
    ax.set_title("Energies")
    calc._log('...energy errors plotted.', toc='energy-plot')

    if plot_forces is True:
        ax = fig.add_subplot(212)

        calc._log('Calculating force errors...', tic='forces')
        force_data = {}
        for hash, image in images.iteritems():
            amp_forces = \
                calc.model.calculate_forces(
                    calc.descriptor.fingerprints[hash],
                    calc.descriptor.fingerprintprimes[hash])
            actual_forces = image.get_forces(apply_constraint=False)
            force_data[hash] = [
                actual_forces,
                abs(np.array(amp_forces) - np.array(actual_forces))]
        calc._log('...force errors calculated.', toc='forces')

        # calculating force rmse
        force_square_error = 0.
        for hash, image in images.iteritems():
            no_of_atoms = len(image)
            for index in range(no_of_atoms):
                for k in range(3):
                    force_square_error += \
                        ((1.0 / 3.0) * force_data[hash][1][index][k] ** 2.) / \
                        no_of_atoms
        force_rmse = np.sqrt(force_square_error / len(images))

        min_act_force = min([force_data[hash][0][index][k]
                            for hash, image in images.iteritems()
                            for index in range(len(image))
                            for k in range(3)])

        max_act_force = max([force_data[hash][0][index][k]
                            for hash, image in images.iteritems()
                            for index in range(len(image))
                            for k in range(3)])

        calc._log('Plotting force errors...', tic='force-plot')
        for hash, image in images.iteritems():
            for index in range(len(image)):
                for k in range(3):
                    ax.plot(force_data[hash][0][index][k],
                            force_data[hash][1][index][k], color)
        # draw horizontal line for rmse
        ax.plot([min_act_force, max_act_force],
                [force_rmse, force_rmse],
                color='black',
                linestyle='dashed',
                lw=1,)
        ax.text(max_act_force,
                force_rmse,
                'force rmse = %5.4f' % force_rmse,
                ha='right',
                va='bottom',
                color='black',)
        ax.set_xlabel("ab initio force, eV/Ang")
        ax.set_ylabel("$|$ab initio force - Amp force$|$")
        ax.set_title("Forces")
        calc._log('...force errors plotted.', toc='force-plot')

    fig.savefig(plotfile)

    if returndata:
        if plot_forces is False:
            return fig, energy_data
        else:
            return fig, energy_data, force_data


def read_trainlog(logfile, verbose=True):
    """Reads the log file from the training process, returning the relevant
    parameters.

    Parameters
    ----------
    logfile : str
        Name or path to the log file.

    verbose : bool
        Write out logfile during analysis.
    """
    data = {}

    with open(logfile, 'r') as f:
        lines = f.read().splitlines()

    def print_(text):
        if verbose:
            print(text)

    # Get number of images.
    for line in lines:
        if 'unique images after hashing.' in line:
            no_images = int(line.split()[0])
            break
    data['no_images'] = no_images

    # Find where convergence data starts.
    startline = None
    for index, line in enumerate(lines):
        if 'Loss function convergence criteria:' in line:
            startline = index
            data['convergence'] = {}
            d = data['convergence']
            break
    else:
        return data

    # Get convergence parameters.
    ready = [False] * 7
    for index, line in enumerate(lines[startline:]):
        if 'energy_rmse:' in line:
            ready[0] = True
            d['energy_rmse'] = float(line.split(':')[-1])
        elif 'force_rmse:' in line:
            ready[1] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_rmse'] = None
                trainforces = False
            else:
                d['force_rmse'] = float(line.split(':')[-1])
                trainforces = True
            print_('train forces: %s' % trainforces)
        elif 'force_coefficient:' in line:
            ready[2] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_coefficient'] = 0.
            else:
                d['force_coefficient'] = float(_)
        elif 'energy_coefficient:' in line:
            ready[3] = True
            d['energy_coefficient'] = float(line.split(':')[-1])
        elif 'energy_maxresid:' in line:
            ready[5] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['energy_maxresid'] = None
            else:
                d['energy_maxresid'] = float(_)
        elif 'force_maxresid:' in line:
            ready[6] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_maxresid'] = None
            else:
                d['force_maxresid'] = float(_)
        elif 'Step' in line and 'Time' in line:
            ready[4] = True
            startline += index + 2
        if ready == [True] * 7:
            break

    for _ in d.iteritems():
        print_('{}: {}'.format(_[0], _[1]))
    E = d['energy_rmse']**2 * no_images
    if trainforces:
        F = d['force_rmse']**2 * no_images
    else:
        F = 0.
    costfxngoal = d['energy_coefficient'] * E + d['force_coefficient'] * F
    d['costfxngoal'] = costfxngoal

    # Extract data (emrs and fmrs are max residuals).
    steps, es, fs, emrs, fmrs, costfxns = [], [], [], [], [], []
    costfxnEs, costfxnFs = [], []
    index = startline
    d['converged'] = None
    while index < len(lines):
        line = lines[index]
        if 'Saving checkpoint data.' in line:
            index += 1
            continue
        elif 'Overwriting file' in line:
            index += 1
            continue
        elif 'optimization completed successfully.' in line:  # old version
            d['converged'] = True
            break
        elif '...optimization successful.' in line:
            d['converged'] = True
            break
        elif 'could not find parameters for the' in line:
            break
        elif '...optimization unsuccessful.' in line:
            d['converged'] = False
            break
        print_(line)
        if trainforces:
            step, time, costfxn, e, _, emr, _, f, _, fmr, _ = line.split()
            fs.append(float(f))
            fmrs.append(float(fmr))
            F = float(f)**2 * no_images
            costfxnFs.append(d['force_coefficient'] * F / float(costfxn))
        else:
            step, time, costfxn, e, _, emr, _ = line.split()
        steps.append(int(step))
        es.append(float(e))
        emrs.append(float(emr))
        costfxns.append(costfxn)
        E = float(e)**2 * no_images
        costfxnEs.append(d['energy_coefficient'] * E / float(costfxn))
        index += 1
    d['steps'] = steps
    d['es'] = es
    d['fs'] = fs
    d['emrs'] = emrs
    d['fmrs'] = fmrs
    d['costfxns'] = costfxns
    d['costfxnEs'] = costfxnEs
    d['costfxnFs'] = costfxnFs

    return data


def plot_convergence(logfile, plotfile='convergence.pdf'):
    """Makes a plot of the convergence of the cost function and its energy
    and force components.

    Parameters
    ----------
    logfile : str
        Name or path to the log file.
    plotfile : str
        Name or path to the plot file.
    """

    data = read_trainlog(logfile)

    # Find if multiple runs contained in data set.
    d = data['convergence']
    steps = range(len(d['steps']))
    breaks = []
    for index, step in enumerate(d['steps'][1:]):
        if step < d['steps'][index]:
            breaks.append(index)

    # Make plots.
    fig = pyplot.figure(figsize=(6., 8.))
    # Margins, vertical gap, and top-to-bottom ratio of figure.
    lm, rm, bm, tm, vg, tb = 0.12, 0.05, 0.08, 0.03, 0.08, 4.
    bottomaxheight = (1. - bm - tm - vg) / (tb + 1.)

    ax = fig.add_axes((lm, bm + bottomaxheight + vg,
                       1. - lm - rm, tb * bottomaxheight))
    ax.semilogy(steps, d['es'], 'b', lw=2, label='energy rmse')
    ax.semilogy(steps, d['emrs'], 'b:', lw=2, label='energy maxresid')
    if d['force_rmse']:
        ax.semilogy(steps, d['fs'], 'g', lw=2, label='force rmse')
        ax.semilogy(steps, d['fmrs'], 'g:', lw=2, label='force maxresid')
    ax.semilogy(steps, d['costfxns'], color='0.5', lw=2,
                label='loss function')
    # Targets.
    if d['energy_rmse']:
        ax.semilogy([steps[0], steps[-1]], [d['energy_rmse']] * 2,
                    color='b', linestyle='-', alpha=0.5)
    if d['energy_maxresid']:
        ax.semilogy([steps[0], steps[-1]], [d['energy_maxresid']] * 2,
                    color='b', linestyle=':', alpha=0.5)
    if d['force_rmse']:
        ax.semilogy([steps[0], steps[-1]], [d['force_rmse']] * 2,
                    color='g', linestyle='-', alpha=0.5)
    if d['force_maxresid']:
        ax.semilogy([steps[0], steps[-1]], [d['force_maxresid']] * 2,
                    color='g', linestyle=':', alpha=0.5)
    ax.set_ylabel('error')
    ax.legend(loc='best', fontsize=9.)
    if len(breaks) > 0:
        ylim = ax.get_ylim()
        for b in breaks:
            ax.plot([b] * 2, ylim, '--k')

    if d['force_rmse']:
        # Loss function component plot.
        axf = fig.add_axes((lm, bm, 1. - lm - rm, bottomaxheight))
        axf.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                         color='blue')
        axf.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                         y2=np.array(d['costfxnEs']) +
                         np.array(d['costfxnFs']),
                         color='green')
        axf.set_ylabel('loss function component')
        axf.set_xlabel('loss function call')
        axf.set_ylim(0, 1)
    else:
        ax.set_xlabel('loss function call')

    fig.savefig(plotfile)
    pyplot.close(fig)
