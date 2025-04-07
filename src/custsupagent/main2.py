from custsupagent.crew1 import RestaurantCrew

Crew = RestaurantCrew()

def kickoffcrew():
    result = Crew.crew().kickoff(inputs={"question": "What is your physical location address?"})
    print(result.raw)