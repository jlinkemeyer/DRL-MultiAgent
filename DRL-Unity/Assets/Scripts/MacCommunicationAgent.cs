using System;
using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.UIElements;
using Random = UnityEngine.Random;

public class MacCommunicationAgent : Agent
{
    public GameObject ground;
    public GameObject area;
    public bool useVectorObs;
    Rigidbody m_AgentRb;
    Renderer m_GroundRenderer;
    MacSettings m_MACSettings;
    int m_Selection;
    private Vector3 communicationVector;

    private Vector3[] blockPositionMemory;
    private Vector3 agentPositionMemory;

    // TODO
    public override void Initialize()
    {
        // Memorize the agent positions for episode reset
        agentPositionMemory = this.transform.position;

        // TODO
        m_MACSettings = FindObjectOfType<MacSettings>();
        m_AgentRb = GetComponent<Rigidbody>();
        m_GroundRenderer = ground.GetComponent<Renderer>();
    }
    
    // TODO
    public override void CollectObservations(VectorSensor sensor)
    {
        if (useVectorObs)
        {
            sensor.AddObservation(StepCount / (float)MaxStep);
        }
    }

    // TODO: The agent can move around in the space
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = act[0];
        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                rotateDir = transform.up * -1f;
                break;
        }
        transform.Rotate(rotateDir, Time.deltaTime * 150f);
        m_AgentRb.AddForce(dirToGo * m_MACSettings.agentRunSpeed, ForceMode.VelocityChange);
    }
    
    public void Talk(ActionSegment<int> act)
    {
        var action = act[1];
        communicationVector = new Vector3(0, 0, 0);
        switch (action)
        {
            case 1:
                communicationVector = new Vector3(0, 0, 0);
                break;
            case 2:
                communicationVector = new Vector3(0, 0, 1);
                break;
            case 3:
                communicationVector = new Vector3(0, 1, 0);
                break;
            case 4:
                communicationVector = new Vector3(1, 0, 0);
                break;
            case 5:
                communicationVector = new Vector3(0, 1, 1);
                break;
            case 6:
                communicationVector = new Vector3(1, 0, 1);
                break;
            case 7:
                communicationVector = new Vector3(1, 1, 0);
                break;
            case 8:
                communicationVector = new Vector3(1, 1, 1);
                break;
        }
    }

    public Vector3 Communicate()
    {
        return communicationVector;
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 4;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
    }
    
    // TODO: On each timestep, the agent receives a small negative reward + performs an action
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        AddReward(-1f / MaxStep);
        MoveAgent(actionBuffers.DiscreteActions);
        Talk(actionBuffers.DiscreteActions);
    }

    // TODO Box is on goal position
    public void ScoredAGoal(Collision col, float score)
    {
        //Give CommunicationAgent Reward
        AddReward(score);
    }

    // TODO: Sets the settings for the episode
    /*public override void OnEpisodeBegin()
    {
        bool nonZero = false;
        int numberGoals = 4;
        for (int i = 0; i < numberGoals; i++)
        {
            // Determine color pattern (0 = no show, 1 = blue, 2 = red)
            int rd = Random.Range(0, 3);
            
            // Make sure that at least one goal is active
            if (rd > 0)
            {
                nonZero = true;
            }
            if (i == numberGoals-1 && !nonZero)
            {
                rd = Random.Range(1, 3);
            }

            if (rd == 0)
            {
                // Do not show goal and instruction object
                goals[i].SetActive(false);
                instructions[i].SetActive(false);
                goals[i].GetComponent<MacGoalWasHit>().SetGoalHit(true); // Set this internally to not run into problems
            } 
            else if (rd == 1)
            {
                goals[i].SetActive(true);
                instructions[i].SetActive(true);
                instructions[i].GetComponent<Renderer>().material = materialBlue;
                goals[i].tag = "GoalBlue";
                goals[i].GetComponent<MacGoalWasHit>().SetGoalHit(false);
                //goals[i].GetComponent<MacDisableBlocks>().DeactivateRed();
            }
            else
            {
                goals[i].SetActive(true);
                instructions[i].SetActive(true);
                instructions[i].GetComponent<Renderer>().material = materialRed;
                goals[i].tag = "GoalRed";
                goals[i].GetComponent<MacGoalWasHit>().SetGoalHit(false);
                //goals[i].GetComponent<MacDisableBlocks>().DeactivateBlue();
            }
            
            goals[i].GetComponent<MacDisableBlocks>().SetActive();
        }
        
        ResetBlocks();

        // TODO: Set initial position, rotation, and velocity
        transform.position = agentPositionMemory;
        transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        m_AgentRb.velocity *= 0f;

        // TODO
        //m_statsRecorder.Add("Goal/Correct", 0, StatAggregationMethod.Sum);
        //m_statsRecorder.Add("Goal/Wrong", 0, StatAggregationMethod.Sum);
    }*/
    
    // TODO: Sets the settings for the episode
    public override void OnEpisodeBegin()
    {

        // TODO: Set initial position, rotation, and velocity
        transform.position = agentPositionMemory;
        transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        m_AgentRb.velocity *= 0f;

        // TODO
        //m_statsRecorder.Add("Goal/Correct", 0, StatAggregationMethod.Sum);
        //m_statsRecorder.Add("Goal/Wrong", 0, StatAggregationMethod.Sum);
    }
    
    void OnCollisionEnter(Collision col)
    {
        // Enforce touching boxes in the beginning
        if (col.gameObject.CompareTag("wall"))
        {
            AddReward(-0.001f);
        }
    }
}


