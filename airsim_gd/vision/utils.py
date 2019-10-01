import airsimneurips as airsim

def setupASClient():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    return client


