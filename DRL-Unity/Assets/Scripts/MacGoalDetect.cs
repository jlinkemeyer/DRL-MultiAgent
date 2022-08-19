//Detect when the orange block has touched the goal.
//Detect when the orange block has touched an obstacle.
//Put this script onto the orange block. There's nothing you need to set in the editor.
//Make sure the goal is tagged with "goal" in the editor.

using System;
using UnityEngine;

public class MacGoalDetect : MonoBehaviour
{
    /// <summary>
    /// The associated agent.
    /// This will be set by the agent script on Initialization.
    /// Don't need to manually set.
    /// </summary>
    [HideInInspector]
    public MacAgent agent;  //
    
    private Material _boxMaterial;
    
    private void Start()
    {
        _boxMaterial = gameObject.GetComponent<Renderer>().material;
    }

    void OnCollisionEnter(Collision col)
    {
        // Box touched a goal
        if (col.gameObject.CompareTag("GoalRed") || col.gameObject.CompareTag("GoalBlue"))
        {
            // Set that the goal was hit (does not matter if correct or wrong)
            col.gameObject.GetComponent<MacGoalWasHit>().SetGoalHit(true);

            // Determine whether goal was correct or wrong
            if ((_boxMaterial.name).Contains("Red") && col.gameObject.CompareTag("GoalRed") ||(_boxMaterial.name).Contains("Blue") && col.gameObject.CompareTag("GoalBlue"))
            {
                // Remove the tag of the goal such that the agent cannot score in that goal again
                col.gameObject.tag = "Untagged";
                // positive reward for correct goal
                agent.ScoredAGoal(1f);
            }
            else
            {
                // Remove the tag of the goal such that the agent cannot score in that goal again
                col.gameObject.tag = "Untagged";
                // negative reward for wrong goal
                agent.ScoredAGoal( -0.1f);
            }
        }
    }
}