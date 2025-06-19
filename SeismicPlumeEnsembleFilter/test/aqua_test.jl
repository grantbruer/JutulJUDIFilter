using Aqua
import SeismicPlumeEnsembleFilter

Aqua.test_all(SeismicPlumeEnsembleFilter, ambiguities=false, piracies=false)
Aqua.test_ambiguities(SeismicPlumeEnsembleFilter)
