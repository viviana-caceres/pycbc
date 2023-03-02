import numpy as np
cimport numpy as cnp
from cython import wraparound, boundscheck, cdivision
from libc.math cimport M_PI, sqrt
from libc.math cimport round as cround


ctypedef fused REALTYPE:
    float
    double


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def findchirp_cluster_over_window_cython\
        (cnp.ndarray[cnp.int32_t, ndim=1] times,
         cnp.ndarray[REALTYPE, ndim=1] absvalues, int window_length,
         cnp.ndarray[cnp.int32_t, ndim=1] indices, int tlen):
    cdef int j = 0
    cdef int curr_ind = 0
    cdef int i
    
    for i in range(tlen):
        if ((times[i] - times[curr_ind]) > window_length):
            j += 1
            indices[j] = i
            curr_ind = i
        elif (absvalues[i] > absvalues[curr_ind]):
            indices[j] = i
            curr_ind = i
    return j


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def logsignalrateinternals_computepsignalbins(
    double[:] pdif,
    double[:] tdif,
    double[:] sdif,
    int[:] pbin,
    int[:] tbin,
    int[:] sbin,
    float[:] p,
    double[:] t,
    float[:] s,
    float[:] sig,
    float[:] pref,
    double[:] tref,
    float[:] sref,
    float[:] sigref,
    double[:] shift,
    long int[:] rtype,
    double sense,
    double senseref,
    double twidth,
    double pwidth,
    double swidth,
    int to_shift_ref,
    int to_shift_ifo,
    int length
):
    cdef:
        int idx, ridx

    for idx in range(length):
        ridx = rtype[idx]
        pdif[idx] = (pref[ridx] - p[ridx]) % (M_PI * 2)
        if pdif[idx] < 0:
            # C modulus operator is not same as python's, correct for this
            pdif[idx] += (M_PI * 2)
        tdif[idx] = shift[ridx] * to_shift_ref + tref[ridx] - shift[ridx] * to_shift_ifo - t[ridx]
        sdif[idx] = (s[ridx] * sense * sqrt(sigref[ridx])) / (sref[ridx] * senseref * sqrt(sig[ridx]))

    for idx in range(length):
        tbin[idx] = <int>(tdif[idx] / twidth)
        pbin[idx] = <int>(pdif[idx] / pwidth)
        sbin[idx] = <int>(sdif[idx] / swidth)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def logsignalrateinternals_compute2detrate(
    int[:] nbinned0,
    int[:] nbinned1,
    int[:] nbinned2,
    long int c0_size,
    long int c1_size,
    long int c2_size,
    float[:] rate,
    long int[:] rtype,
    float[:] sref,
    float[:,:,::1] two_det_weights, # This declares a C-contiguous array
    float max_penalty,
    float ref_snr,
    int length
):
    cdef:
        int idx, ridx, id0, id1, id2
        float rescale_fac

    for idx in range(length):
        ridx = rtype[idx]
        id0 = nbinned0[idx] + c0_size / 2
        id1 = nbinned1[idx] + c1_size / 2
        id2 = nbinned2[idx] + c2_size / 2

        # For bins that exist in the signal pdf histogram, apply that pdf value,
        # otherwise apply the "max penalty" value.
        # The bins are specified by 3 indexes (corresponding to the time
        # difference, phase difference and relative sensitivity dimensions).
        if (id0 > 0) and (id0 < c0_size) and (id1 > 0) and (id1 < c1_size) and (id2 > 0) and (id2 < c2_size):
            rate[ridx] = two_det_weights[id0, id1, id2]
        else:
            rate[ridx] = max_penalty
        # Scale by signal population SNR
        rescale_fac = ref_snr / sref[ridx]
        rate[ridx] *= (rescale_fac*rescale_fac*rescale_fac*rescale_fac)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def timecoincidence_constructfold(
    double[:] fold1,
    double[:] fold2,
    double[:] t1,
    double[:] t2,
    double slide_step,
    int length1,
    int length2
):
    cdef:
        int idx

    for idx in range(length1):
        fold1[idx] = t1[idx] % slide_step

    for idx in range(length2):
        fold2[idx] = t2[idx] % slide_step


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def timecoincidence_findidxlen(
    long int[:] left,
    long int[:] right,
    int leftlength,
):
    cdef:
        int idx, tslength

    tslength = 0
    for idx in range(leftlength):
        tslength += right[idx] - left[idx]
    return tslength


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def timecoincidence_constructidxs(
    unsigned int[:] idx1,
    unsigned int[:] idx2,
    long int[:] sort1,
    long int[:] sort2,
    long int[:] left,
    long int[:] right,
    int leftlength,
    int sort2length
):
    cdef:
        int idx, jdx, count, currlen

    # Construct sort1
    count = 0
    for idx in range(leftlength):
        currlen = right[idx] - left[idx]
        for jdx in range(currlen):
            idx1[count] = sort1[idx]
            count += 1

    # Construct sort2
    count = 0
    for idx in range(leftlength):
        for jdx in range(left[idx], right[idx]):
            idx2[count] = sort2[jdx % sort2length]
            count += 1


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def timecoincidence_getslideint(
    int [:] slide,
    double[:] t1,
    double[:] t2,
    unsigned int[:] idx1,
    unsigned int[:] idx2,
    double slide_step
):
    cdef:
        int idx, length

    length = idx1.shape[0]

    for idx in range(length):
        diff = (t1[idx1[idx]] - t2[idx2[idx]]) / slide_step
        slide[idx] = <int>(cround(diff))


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def coincbuffer_expireelements(
    float[:] cbuffer,
    int[:] timer1,
    int[:] timer2,
    int time1,
    int time2,
    int expiration,
    int length,
):
    cdef:
        int idx, keep_count

    keep_count = 0
    for idx in range(length):
        if (timer1[idx] >= (time1 - expiration)) and (timer2[idx] >= (time2 - expiration)):
            cbuffer[keep_count] = cbuffer[idx]
            timer1[keep_count] = timer1[idx]
            timer2[keep_count] = timer2[idx]
            keep_count += 1

    return keep_count


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def coincbuffer_numgreater(
    float[:] cbuffer,
    int length,
    float value
):
    cdef:
        int idx, count

    count = 0
    for idx in range(length):
        count += cbuffer[idx] > value
    return count


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def timecluster_cython(
    unsigned int[:] indices,
    long int[:] left,
    long int[:] right,
    REALTYPE[:] stat,
    int leftlen,
):
    cdef:
        int i, j, k, max_loc
        long int l, r
        REALTYPE max_val

    # i is the index we are inspecting, j is the next one to save
    i = 0
    j = 0
    while i < leftlen:
        l = left[i]
        r = right[i]

        # If there are no other points to compare it is obviously the max
        if (r - l) == 1:
            indices[j] = i
            j += 1
            i += 1
            continue

        # Find the location of the maximum within the time interval around i
        # Following block replaces max_loc = argmax(stat[l:r]) + l
        max_val = stat[l]
        max_loc = l
        for k in range(l + 1, r):
            if stat[k] > max_val:
                max_val = stat[k]
                max_loc = k

        # If this point is the max, we can skip to the right boundary
        if max_loc == i:
            indices[j] = i
            i = r
            j += 1

        # If the max is later than i, we can skip to it
        elif max_loc > i:
            i = max_loc

        elif max_loc < i:
            i += 1
    return j


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def pynu_timecoincidence_findidxlen(
    double[:] fold1,
    double[:] fold2,
    double delay,
    double slop,
):
    cdef:
        int idx, tslength
        cnp.ndarray[cnp.double_t, ndim=1] correct_values

    # Initialize the correct_values array with zeros
    correct_values = np.zeros(len(fold1))

    # We need to find what values would correspond to the given
    # delay for each trigger in fold1
    for idx in range(len(fold1)):
        correct_values[idx] = fold1[idx] + delay

    tslength = 0
    for correct_value in correct_values:
        # We find how many triggers in fold2 have the correct
        # value to coincide with each trigger in fold1
        for idx in range(len(fold2)): # This can later be changed to include an error
            if abs(fold2[idx]-correct_value) <= slop:
                tslength += 1

    return tslength


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def pynu_timecoincidence_constructidxs(
    cnp.ndarray[cnp.uint32_t, ndim=1] idx1,
    cnp.ndarray[cnp.uint32_t, ndim=1] idx2,
    long int[:] sort1,
    long int[:] sort2,
    double[:] fold1,
    double[:] fold2,
    double slide_step,
    double delay,
    double slop
):
    cdef:
        int idx, jdx, count
        unsigned int[:] indices_of_correct_delays

    # Construct sort1
    count = 0
    # For each trigger in fold1...
    for idx in range(len(fold1)):
        # We find where fold2 is equal to fold1 + delay
        indices_of_correct_delays = np.array([], dtype=np.uint32)
        for jdx in range(len(fold2)):
            if abs(fold2[jdx] - fold1[idx] - delay) <= slop:
                indices_of_correct_delays = np.append(indices_of_correct_delays, np.uint32(jdx))

        if slide_step:
        # Since fold2 is three times the length it should be, I
        # need to change the indices
            indices_of_correct_delays %= int(len(fold2)/3)

        # Now we fill up our array with the indices of the correct delays
        for jdx in indices_of_correct_delays:

            # First we fill up idx1 with the index of the current trigger we are analyzing
            idx1[count] = sort1[idx]

            # Then we fill up idx2 with the index of the coinc triggers with the right delay
            idx2[count] = sort2[jdx]
            count += 1

    # Now we can sort our results
    idx2 = idx2[np.argsort(idx1)]
    idx1 = np.sort(idx1)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def pynu_timecoincidence_getslideint(
    int [:] slide,
    double[:] t1,
    double[:] t2,
    unsigned int[:] idx1,
    unsigned int[:] idx2,
    double delay,
    double slide_step
):
    cdef:
        int idx, length

    length = idx1.shape[0]

    for idx in range(length):
        diff = (t1[idx1[idx]] - t2[idx2[idx]] + delay) / slide_step
        slide[idx] = <int>(cround(diff))

