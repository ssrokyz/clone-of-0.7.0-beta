import os
import shutil
import tempfile
import warnings


def save_to_prophet(calc, filename='potential_', overwrite=False,
                    units="metal"):
    """Saves the calculator in a way that it can be used with PROPhet.

    Parameters
    ----------
    calc : obj
        A trained Amp calculator object.
    filename : str
        File object or path to the file to write to.
    overwrite : bool
        If an output file with the same name exists, overwrite it.
    units : str
        LAMMPS units style to be used with the outfile file.
    """

    from ase.calculators.lammpslib import unit_convert

    if os.path.exists(filename):
        if overwrite is False:
            oldfilename = filename
            filename = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                   suffix='.amp').name
            calc._log('File "%s" exists. Instead saving to "%s".' %
                      (oldfilename, filename))
        else:
            oldfilename = tempfile.NamedTemporaryFile(mode='w',
                                                      delete=False,
                                                      suffix='.amp').name

            calc._log('Overwriting file: "%s". Moving original to "%s".'
                      % (filename, oldfilename))
            shutil.move(filename, oldfilename)

    desc_pars = calc.descriptor.parameters
    model_pars = calc.model.parameters
    if (desc_pars['mode'] != 'atom-centered' or
       model_pars['mode'] != 'atom-centered'):
        raise NotImplementedError(
            'PROPhet requires atom-centered symmetry functions.')
    if desc_pars['cutoff']['name'] != 'Cosine':
        raise NotImplementedError(
            'PROPhet requires cosine cutoff functions.')
    if model_pars['activation'] != 'tanh':
        raise NotImplementedError(
            'PROPhet requires tanh activation functions.')
    els = desc_pars['elements']
    n_els = len(els)
    length_G2 = int(n_els)
    length_G4 = int(n_els*(n_els+1)/2)
    cutoff = (desc_pars['cutoff']['kwargs']['Rc'] /
              unit_convert('distance', units))
    # Get correct order of elements listed in the Amp object
    el = desc_pars['elements'][0]
    n_G2 = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G2')
    n_G4 = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G4')
    els_ordered = []
    if n_G2 > 0:
        for Gs in range(n_els):
            els_ordered.append(desc_pars['Gs'][el][Gs]['element'])
    elif n_G4 > 0:
        for Gs in range(n_els):
            els_ordered.append(desc_pars['Gs'][el][Gs]['elements'][0])
    else:
        raise RuntimeError('There must be at least one G2 or G4 symmetry '
                           'function.')
    # Write each element's PROPhet input file
    for el in desc_pars['elements']:
        f = open(filename + el, 'w')
        # Write header.
        f.write('nn\n')
        f.write('structure\n')
        # Write elements.
        f.write(el + ':  ')
        for el_i in els_ordered:
            f.write(el_i+' ')
        f.write('\n')
        n_G2_el = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G2')
        n_G4_el = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G4')
        if n_G2_el != n_G2 or n_G4_el != n_G4:
            raise NotImplementedError(
                'PROPhet requires each element to have the same number of '
                'symmetry functions.')
        f.write(str(int(n_G2/length_G2+n_G4/length_G4))+'\n')
        # Write G2s.
        for Gs in range(0, n_G2, length_G2):
            eta = desc_pars['Gs'][el][Gs]['eta']
            if (eta > 10):
                warnings.warn(
                    'Conversion from Amp to PROPhet leads to energies and '
                    'forces being calculated correctly to within machine '
                    'precision. With the chosen eta of ' + str(eta) + ' '
                    'being greater than 10, it is possible that the '
                    'results of the two codes will not be equal, so the '
                    'neural net should not be used with both codes.'
                    'Please lower the eta values.')
            for i in range(length_G2):
                eta_2 = desc_pars['Gs'][el][Gs+i]['eta']
                if eta != eta_2:
                    raise NotImplementedError(
                        'PROPhet requires each G2 function to have the '
                        'same eta value for all element pairs.')
            f.write('G2 ' + str(cutoff) + ' 0 ' + str(eta/cutoff**2) +
                    ' 0\n')
        # Write G4s (G3s in PROPhet).
        for Gs in range(n_G2, n_G2+n_G4, length_G4):
            eta = desc_pars['Gs'][el][Gs]['eta']
            if (eta > 10):
                warnings.warn(
                    'Conversion from Amp to PROPhet leads to energies and '
                    'forces being calculated correctly to within machine '
                    'precision. With the chosen eta of ' + str(eta) + ' '
                    'being greater than 10, it is possible that the '
                    'results of the two codes will not be equal, so the '
                    'neural net should not be used with both codes.'
                    'Please lower the eta values.')
            gamma = desc_pars['Gs'][el][Gs]['gamma']
            zeta = desc_pars['Gs'][el][Gs]['zeta']
            for i in range(length_G4):
                eta_2 = desc_pars['Gs'][el][Gs+i]['eta']
                gamma_2 = desc_pars['Gs'][el][Gs+i]['gamma']
                zeta_2 = desc_pars['Gs'][el][Gs+i]['zeta']
                if eta != eta_2 or gamma != gamma_2 or zeta != zeta_2:
                    raise NotImplementedError(
                        'PROPhet requires each G4 function to have the '
                        'same eta, gamma, and zeta values for all '
                        'element pairs.')
            f.write('G3 ' + str(cutoff) + ' 0 ' + str(eta/cutoff**2) +
                    ' ' + str(int(zeta)) + ' ' + str(int(gamma)) + '\n')
        # Write input means for G2.
        for i in range(n_els):
            for Gs in range(0, n_G2, length_G2):
                # For debugging, to see the order of the PROPhet file
                # if el==desc_pars['elements'][0]:
                #    print(desc_pars['Gs'][el][Gs+i])
                mean = (model_pars['fprange'][el][Gs+i][1] +
                        model_pars['fprange'][el][Gs+i][0]) / 2.
                f.write(str(mean) + ' ')
        # Write input means for G4.
        for i in range(n_els):
            for j in range(n_els-i):
                for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                    # For debugging, to see the order of the PROPhet file
                    # if el==desc_pars['elements'][0]:
                    #    print(desc_pars['Gs'][el][Gs+j+n_els*i+int((i-i**2)/2)])
                    mean = (model_pars['fprange'][el][Gs + j + n_els * i +
                                                      int((i - i**2) / 2)][1] +
                            model_pars['fprange'][el][Gs + j + n_els * i +
                                                      int((i - i**2) / 2)][0])
                    # NB the G4 mean is doubled to correct for PROPhet
                    # counting each neighbor pair twice as much as Amp
                    f.write(str(mean) + ' ')
        f.write('\n')
        # Write input variances for G2.
        for i in range(n_els):
            for Gs in range(0, n_G2, length_G2):
                variance = (model_pars['fprange'][el][Gs+i][1] -
                            model_pars['fprange'][el][Gs+i][0]) / 2.
                f.write(str(variance) + ' ')
        # Write input variances for G4.
        for i in range(n_els):
            for j in range(n_els-i):
                for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                    variance = (model_pars['fprange'][el][Gs + j + n_els * i +
                                                          int((i - i**2) /
                                                              2)][1] -
                                model_pars['fprange'][el][Gs + j + n_els * i +
                                                          int((i - i**2) /
                                                              2)][0])
                    # NB the G4 variance is doubled to correct for PROPhet
                    # counting each neighbor pair twice as much as Amp
                    f.write(str(variance) + ' ')
        f.write('\n')
        f.write('energy\n')
        # Write output mean.
        f.write('0\n')
        # Write output variance.
        f.write('1\n')
        curr_node = 0
        # Write NN layer architecture.
        for nodes in model_pars['hiddenlayers'][el]:
            f.write(str(nodes)+' ')
        f.write('1\n')
        # Write first hidden layer of the NN for the symmetry functions.
        layer = 0
        f.write('[[ layer ' + str(layer) + ' ]]\n')
        for node in range(model_pars['hiddenlayers'][el][layer]):
            # Write each node of the layer.
            f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
            f.write('   ')
            # G2
            for i in range(n_els):
                for Gs in range(0, n_G2, length_G2):
                    f.write(str(model_pars['weights'][el]
                                [layer + 1][Gs + i][node]))
                    f.write('     ')
            # G4
            for i in range(n_els):
                for j in range(n_els-i):
                    for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                        f.write(str(model_pars['weights'][el]
                                    [layer + 1][Gs + j + n_els * i +
                                                int((i - i**2) / 2)][node]))
                        f.write('     ')
            f.write('\n')
            f.write('   ')
            f.write(str(model_pars['weights'][el][layer+1][-1][node]))
            f.write('\n')
            curr_node += 1
        # Write remaining hidden layers of the NN.
        for layer in range(1, len(model_pars['hiddenlayers'][el])):
            f.write('[[ layer ' + str(layer) + ' ]]\n')
            for node in range(model_pars['hiddenlayers'][el][layer]):
                f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
                f.write('   ')
                for i in range(len(model_pars['weights'][el][layer+1])-1):
                    f.write(str(model_pars['weights'][el][layer+1][i][node]))
                    f.write('     ')
                f.write('\n')
                f.write('   ')
                f.write(str(model_pars['weights'][el][layer+1][-1][node]))
                f.write('\n')
                curr_node += 1
        # Write output layer of the NN, consisting of an activated node.
        f.write('[[ layer ' + str(layer+1) + ' ]]\n')
        f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
        f.write('   ')
        for i in range(len(model_pars['weights'][el][layer+2])-1):
            f.write(str(model_pars['weights'][el][layer+2][i][0]))
            f.write('     ')
        f.write('\n')
        f.write('   ')
        f.write(str(model_pars['weights'][el][layer+2][-1][0]))
        f.write('\n')
        curr_node += 1
        # Write output layer of the NN, consisting of a linear node,
        # representing Amp's scaling.
        f.write('[[ layer ' + str(layer+2) + ' ]]\n')
        f.write('  [ node ' + str(curr_node) + ' ]  linear\n')
        f.write('   ')
        f.write(str(model_pars['scalings'][el]['slope'] /
                    unit_convert('energy', units)))
        f.write('\n')
        f.write('   ')
        f.write(str(model_pars['scalings'][el]['intercept'] /
                    unit_convert('energy', units)))
        f.write('\n')
        f.close()
