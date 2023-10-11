#### Example of what you can do with the --extra files:
# Modify as desired by uncommenting relevant lines starting with "info."

#### Options related to the parameters.
# Use this to derive a new parameter.
# The new parameter can be defined as a function of one or more existing
# parameters. The derived parameter can be included for further derived
# parameters if in the correct order
#info.to_derive = {'der1':'H0*H0*omega_cdm','der2':'der1*H0'}

# Use this to re-order the parameters
# The parameters in this list will be plotted first, using the order provided.
# Afterwards (if not excluded by to_plot) the other parameters are plotted
# in the order they appear in the .param file.
#info.to_reorder = ['der2','H0']

# Use this to rename a parameter (e.g. to make it look better in the labels).
# If you don't use dollars '$...$' the code will try automatically
# to convert your entry to latex (intepreting _, ^, greek letters...)
# but if you use '$...$' with plain latex in between, it will leave
# your input unchanged.
# This can be a little tricky to use and may take a few tries to get right.
# A couple of rules of thumb:
# 1) The name to change is always the exact name in the log.param.
# 2) info will automatically try to convert common math symbols into
# texed symbols. However, not all cases is covered and it is possible
# to manually pass names in latex format. Unfortunately, this sometimes
# conflicts with the code and the outcome is not as desired.
#info.to_change = {'rs_d': 'r^s_d', 'omega_cdm': '$\Omega_\mathrm{m}$'}

# Use this to change the scale factor normalising the parameters.
# This has problems with some parameter names where a number is
# included in the name, e.g. '100*theta_s'.
#info.new_scales = {'r^s_d': 100}

# Use this to plot just a selection of parameters (if you have
# changed the names with 'info.to_change', you must put the new names here!)
#info.to_plot = ['omega_b', 'Omega_m', 'H0', 'Omega_Lambda', 'r^s_d']

# Use this to control the boundaries of 1d and 2d plots.
# Note if your log.param contains parameter boundaries the 1d plot will not
# exceed those boundaries even if the parameter was redefined. You may need
# to modify the log.param in such cases, even though that is usually a bad idea.
# The ticks at the limits are placed at Xmin + 0.1 deltaX and Xmax - 0.1 deltaX,
# i.e. in the example below for H0 at 61 and 69.
#info.force_limits = {'H0':[60,70], 'z_reio':[5,15]}


#### Customize appearance of contours.
# You may actually even redefine the colors
# (each pair stands for [95% contour color,68% contour color]).
#info.MP_color = {'Red':['#E37C80','#CE121F'],'Blue':['#7A98F6','#1157EF'],'Green':['#88B27A','#297C09'],'Orange':['#F3BE82','#ED920F'],'Grey':['#ABABAB','#737373'],'Purple':['#B87294','#88004C']}

# If you want to change the order of colors
# (same order as for legendnames).
#info.MP_color_cycle = [info.MP_color['Green'], info.MP_color['Orange'], info.MP_color['Blue']]

# Adjust the transparency of the lines and filled contours
# (same order as for legendnames).
#info.alphas = [1.0, 0.8, 0.6, 0.4, 0.2]

# Set line width for 1d and 2d plots (minimum 2 recommended).
#info.line_width = 4


#### Use this to customise the ticks, labels and legends.
# Number of ticks per axis, size of ticks, number of decimal places on ticks.
#info.ticknumber = 5
#info.ticksize = 10
#info.decimal = 3

# Fontsize applies to axis labels and title, legendsize to legend.
#info.fontsize = 16
#info.legendsize = 16


#### Further customize legends.
# Decide whether to plot a legend or not
# (if not None, the legend will be added if
# there is more than one directory to analyse).
#info.plot_legend_1d = True
#info.plot_legend_2d = False

# Use this to customise the legend
# (one array entry for each plotted directory).
# The order here refers to the order in which you pass
# the directories to analyse.
#info.legendnames = ['Hubble']

# Legend type, to choose between 'top' (previous style) and 'sides' (new style).
# It modifies the place where the name of the variable appear.
#info.legend_style = 'sides'


#### Advanced plotting options to manipulate appearance of contours.
# Number of bins in the histogram, lower for smoother contours,
# but extra modes or local maxima may get washed out. Sometimes
# useful if the code struggles to compute confidence intervals.
#info.bins = 20

# Width of gaussian smoothing for plotting posteriors in units
# of bin size, increase for smoother appearance.
# If non-Gaussian posteriors recommend setting to 0.35
#info.gaussian_smoothing = 0.5

# Interpolation factor for plotting posteriors, 1 means no interpolation,
# increase for smoother contours (integer).
# If non-Gaussian posteriors recommend setting to 2
#info.interpolation_smoothing = 4

# Smoothing scheme for 1d posteriors, 0 means no smoothing, 1 means cubic
# interpolation, higher means fitting ln(L) with polynomial of order n (integer).
# IF NON-GAUSSIAN POSTERIORS TURN THIS OFF (set to 0)
# There are recommended lines to flip in montepython/analyze.py to make
# plots nicer in this case. Search for posterior_smoothing
#info.posterior_smoothing = 5


#### Add extra features to the plot via python scripts.
# Add list of python scripts for customisation of 1d or 2d plots,
# that are executed before plotting the probability lines or contours.
# E.g. you can add a band showing an H0 measurement or any other custom contours.
#info.custom1d = []
#info.custom2d = ['add_h_contour.py','add_sigma8_Omegam_contour.py']


#### Old feature:
# Use this to replace one column with a new value,
# defined as a function of one or more existing parameters.
# Note that any parameter redefined will have those changes included
# in subsequent parameter redefinitions. The order changes are applied
# is the order in the log.param, NOT the order in this dictionary!
# This will also modify the bestfit file in a possibly unsafe way.
#info.redefine = {'omega_cdm': '(0.01*omega_b+omega_cdm)/(H0/100.)**2'}


# Any other lines of plain python can be written here without special
# formatting, they will be executed as extra lines of codes, but only
# at a precise point in the initialisation of the "Information" class.
