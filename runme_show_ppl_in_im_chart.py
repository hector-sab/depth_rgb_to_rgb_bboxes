import utils as ut

if __name__=='__main__':
	dir_ = '/data/HectorSanchez/database/PeopleCounter/camara1/'
	path = dir_+'00000123/'

	sims = ut.FindPeople()
	sims.runme(path,alpha=0.98)
	sims.chart()