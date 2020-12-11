import argparse
import mdpSolver
import mdp

if __name__ == '__main__':
    # argparse object
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp",type=str)
    parser.add_argument("--algorithm",type=str)
    args = parser.parse_args()

    # generate mdp instance
    mdp_instance = mdp.MDP(args.mdp)
    # solve
    if args.algorithm == 'vi':
        mdpSolver.value_iteration(mdp_instance)
    elif args.algorithm == 'hpi':
        mdpSolver.howard_pi(mdp_instance)
    elif args.algorithm == 'lp':
        mdpSolver.lp(mdp_instance)
    else:
        print("unknown algorithm")
    
    print(mdp_instance.prettyPrint())