#ifndef SPIKEKERNELS_H_INCLUDED
#define SPIKEKERNELS_H_INCLUDED

#define SPIKE 1.0f

template <class T>
__global__ void getSpikesKernel(
	T* __restrict__ d_s,
	T* __restrict__ d_u,
	unsigned nNeurons, unsigned Ns, 
	float theta)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(neuronID >= nNeurons)	return;
	unsigned i = 0;
	{
		unsigned linearID = i + neuronID * Ns;
		if(d_u[linearID] >= theta)
		{
			d_s[linearID] = SPIKE;
			d_u[linearID] -= theta;
		}
		else	d_s[linearID] = 0.0f;
	}
	for(i=1; i<Ns; ++i)
	{
		unsigned linearID = i + neuronID * Ns;
		d_u[linearID] += d_u[linearID-1];
		if(d_u[linearID] >= theta)
		{
			d_s[linearID] = SPIKE;
			d_u[linearID] -= theta;
		}
		else	d_s[linearID] = 0.0f;
	}
}

template <class T>
void getSpikes(T* d_s, T* d_u, unsigned nNeurons, unsigned Ns, float theta)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	getSpikesKernel<T><<< block, thread >>>(d_s, d_u, nNeurons, Ns, theta);
}

#endif // SPIKEKERNELS_H_INCLUDED