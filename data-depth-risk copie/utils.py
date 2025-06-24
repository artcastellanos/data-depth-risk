def gen_name(begin='',end='',ndepth=None,nalgo=None,n=None,nproj=None,sigma=None,num_z=None,nruns=None,d=None,r=None,radius=None,percentKDE=None,distrib=None,df=None,seed_num=None,beta=None):

    path = begin
    if n is not None:
        path += "_n_" +  str(n)
    if nruns is not None:
        path += "_nruns_" +  str(nruns)
    if sigma is not None:
        path += "_sigma_"+ str(sigma)
    if r is not None:
        path += "_r_" + str(r)
    if d is not None:
        path += "_d_" + str(d)       
    if radius is not None:
        path += "_radius_" + str(radius)
    if distrib is not None:
        path += "_distrib_" + distrib
    if seed_num is not None:
        path += "_seed_" + str(seed_num)
    path += end 

    return path


