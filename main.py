from src.nested import BayesFactor
import config as CONFIG

TEST = BayesFactor(0.2, CONFIG.NDATA)
results = TEST.summaries(5000, verbose=False, save=True)
