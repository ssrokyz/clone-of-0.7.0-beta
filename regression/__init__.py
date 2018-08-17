from ..utilities import ConvergenceOccurred


class Regressor:

    """Class to manage the regression of a generic model. That is, for a
    given parameter set, calculates the cost function (the difference in
    predicted energies and actual energies across training images), then
    decides how to adjust the parameters to reduce this cost function.
    Global optimization conditioners (e.g., simulated annealing, etc.) can
    be built into this class.

    Parameters
    ----------
    optimizer : str
        The optimizer to use. Several defaults are available including
        'L-BFGS-B', 'BFGS', 'TNC', or 'NCG'.  Alternatively, any function can
        be supplied which behaves like scipy.optimize.fmin_bfgs.
    optimizer_kwargs : dict
        Optional keywords for the corresponding optimizer.
    lossprime : boolean
        Decides whether or not the regressor needs to be fed in by gradient of
        the loss function as well as the loss function itself.
    """

    def __init__(self, optimizer='BFGS', optimizer_kwargs=None,
                 lossprime=True):
        """optimizer can be specified; it should behave like a
        scipy.optimize optimizer. That is, it should take as its first two
        arguments the function to be optimized and the initial guess of the
        optimal paramters. Additional keyword arguments can be fed through
        the optimizer_kwargs dictionary."""

        user_kwargs = optimizer_kwargs
        optimizer_kwargs = {}
        if optimizer == 'BFGS':
            from scipy.optimize import fmin_bfgs as optimizer
            optimizer_kwargs = {'gtol': 1e-15, }
        elif optimizer == 'L-BFGS-B':
            from scipy.optimize import fmin_l_bfgs_b as optimizer
            optimizer_kwargs = {'factr': 1e+02,
                                'pgtol': 1e-08,
                                'maxfun': 1000000,
                                'maxiter': 1000000}
            import scipy
            from distutils.version import StrictVersion
            if StrictVersion(scipy.__version__) >= StrictVersion('0.17.0'):
                optimizer_kwargs['maxls'] = 2000
        elif optimizer == 'TNC':
            from scipy.optimize import fmin_tnc as optimizer
            optimizer_kwargs = {'ftol': 0.,
                                'xtol': 0.,
                                'pgtol': 1e-08,
                                'maxfun': 1000000, }
        elif optimizer == 'NCG':
            from scipy.optimize import fmin_ncg as optimizer
            optimizer_kwargs = {'avextol': 1e-15, }

        if user_kwargs:
            optimizer_kwargs.update(user_kwargs)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lossprime = lossprime

    def regress(self, model, log):
        """Performs the regression. Calls model.get_loss,
        which should return the current value of the loss function
        until convergence has been reached, at which point it should
        raise a amp.utilities.ConvergenceException.

        Parameters
        ----------
        model : object
            Class representing the regression model.
        log : str
            Name of script to log progress.
        """
        log('Starting parameter optimization.', tic='opt')
        log(' Optimizer: %s' % self.optimizer)
        log(' Optimizer kwargs: %s' % self.optimizer_kwargs)
        x0 = model.vector.copy()
        try:
            if self.lossprime:
                self.optimizer(model.get_loss, x0, model.get_lossprime,
                               **self.optimizer_kwargs)
            else:
                self.optimizer(model.get_loss, x0, **self.optimizer_kwargs)

        except ConvergenceOccurred:
            log('...optimization successful.', toc='opt')
            return True
        else:
            log('...optimization unsuccessful.', toc='opt')
            if self.lossprime:
                max_lossprime = \
                    max(abs(max(model.lossfunction.dloss_dparameters)),
                        abs(min(model.lossfunction.dloss_dparameters)))
                log('...maximum absolute value of loss prime: %.3e'
                    % max_lossprime)
            return False
