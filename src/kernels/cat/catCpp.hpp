#ifndef CATCPP_HPP_INCLUDED
#define CATCPP_HPP_INCLUDED

#define SPIKE 1.0f

template <class T>
void getSpikesKernel(
	T* d_s,
	T* d_u,
	unsigned nNeurons, unsigned Ns, 
	float theta, int neuronID)
{
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
	for(int i = 0; i < nNeurons; i++)
	getSpikesKernel<T>(d_s, d_u, nNeurons, Ns, theta, i);
}

#endif 