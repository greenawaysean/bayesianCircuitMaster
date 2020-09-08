from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(group='samsung', project='imperial')

backends = provider.backends()
operational = [i.status().operational for i in backends]
num_jobs = [i.status().pending_jobs for i in backends]

backend = backends[0]
print(backend.properties)
