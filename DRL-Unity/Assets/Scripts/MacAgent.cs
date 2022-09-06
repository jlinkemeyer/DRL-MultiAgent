using System;
using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.UIElements;
using Random = UnityEngine.Random;

public class MacAgent : Agent
{
    public GameObject ground;
    public GameObject area;
    public GameObject goalA;
    public GameObject goalB;
    public GameObject goalC;
    public GameObject goalD;
    public GameObject instructionA;
    public GameObject instructionB;
    public GameObject instructionC;
    public GameObject instructionD;
    public Material materialRed;
    public Material materialBlue;
    public GameObject[] blocks;
    public bool useVectorObs;
    Rigidbody m_AgentRb;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    MacSettings m_MACSettings;
    int m_Selection;
    StatsRecorder m_statsRecorder;
    
    [HideInInspector]
    public MacGoalDetect macGoalDetect;

    private Vector3[] blockPositionMemory;
    private Vector3 agentPositionMemory;
    private GameObject[] goals;
    private GameObject[] instructions;

    // TODO
    public override void Initialize()
    {
        // Memorize the original block and agent positions for episode reset
        blockPositionMemory = new Vector3[blocks.Length];
        for (int i = 0; i < blocks.Length; i++)
        {
            blockPositionMemory[i] = blocks[i].transform.position;
            macGoalDetect = blocks[i].GetComponent<MacGoalDetect>();
            macGoalDetect.agent = this;
        }
        agentPositionMemory = this.transform.position;
        
        // Get all goals and instructions
        goals = new GameObject[] {goalA, goalB, goalC, goalD};
        instructions = new GameObject[] {instructionA, instructionB, instructionC, instructionD};

        // TODO
        m_MACSettings = FindObjectOfType<MacSettings>();
        m_AgentRb = GetComponent<Rigidbody>();
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
        m_statsRecorder = Academy.Instance.StatsRecorder;
    }
    
    // TODO
    public override void CollectObservations(VectorSensor sensor)
    {
        if (useVectorObs)
        {
            sensor.AddObservation(StepCount / (float)MaxStep);
        }
    }

    // TODO: The ground material changes upon success or failure of the agent
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time);
        m_GroundRenderer.material = m_GroundMaterial;
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
    }

    // TODO Box is on goal position
    public void ScoredAGoal(Collision col, float score)
    {
        // Check if episode is finished (when all goals received one object)
        bool done = true;
        for (int i = 0; i < goals.Length; i++)
        {
            if (!goals[i].GetComponent<MacGoalWasHit>().GetGoalHit())
            {
                done = false;
                break;
            }
        }

        // TODO Disable Rigidbody to disable movability of the block?
        col.gameObject.SetActive(false);
        col.gameObject.GetComponent<MacDisableBlocks>().SetActive();

        //Give Agent Reward
        AddReward(score);

        // Swap ground material for a bit to indicate we scored -> red for wrong combination, green for correct
        if (score < 0)
        {
            StartCoroutine(GoalScoredSwapGroundMaterial(m_MACSettings.failMaterial, 0.5f));
        }
        else
        {
            StartCoroutine(GoalScoredSwapGroundMaterial(m_MACSettings.goalScoredMaterial, 0.5f));
        }

        if (done)
        {
            EndEpisode();
        }
    }

    void ResetBlocks()
    {
        for (int i = 0; i < blocks.Length; i++)
        {
            blocks[i].transform.position = blockPositionMemory[i];
            blocks[i].GetComponent<Rigidbody>().velocity *= 0;
        }
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
                instructions[i].tag = "instructionBlue";
                goals[i].GetComponent<MacCustomTag>().UpdateCustomTag("GoalBlue");
                goals[i].GetComponent<MacGoalWasHit>().SetGoalHit(false);
                //goals[i].GetComponent<MacDisableBlocks>().DeactivateRed();
            }
            else
            {
                goals[i].SetActive(true);
                instructions[i].SetActive(true);
                instructions[i].GetComponent<Renderer>().material = materialRed;
                instructions[i].tag = "instructionRed";
                goals[i].GetComponent<MacCustomTag>().UpdateCustomTag("GoalRed");
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
        // Get a random goal:
        int rdGoal = 1; Random.Range(0, 4);
        int numberGoals = 4;
        for (int i = 0; i < numberGoals; i++)
        {
            // Determine color pattern (0 = no show, 1 = blue, 2 = red)
            int rd = Random.Range(1, 3);

            if (rd == 1)
            {
                goals[i].SetActive(true);
                instructions[i].SetActive(true);
                instructions[i].GetComponent<Renderer>().material = materialBlue;
                instructions[i].tag = "instructionBlue";
                goals[i].GetComponent<MacCustomTag>().UpdateCustomTag("GoalBlue");
                goals[i].GetComponent<MacGoalWasHit>().SetGoalHit(false);
                //goals[i].GetComponent<MacDisableBlocks>().DeactivateRed();
            }
            else
            {
                goals[i].SetActive(true);
                instructions[i].SetActive(true);
                instructions[i].GetComponent<Renderer>().material = materialRed;
                instructions[i].tag = "instructionRed";
                goals[i].GetComponent<MacCustomTag>().UpdateCustomTag("GoalRed");
                goals[i].GetComponent<MacGoalWasHit>().SetGoalHit(false);
                //goals[i].GetComponent<MacDisableBlocks>().DeactivateBlue();
            }
            
            if (i != rdGoal)
            {
                // Do not show goal and instruction object
                goals[i].SetActive(false);
                instructions[i].SetActive(false);
                goals[i].GetComponent<MacGoalWasHit>().SetGoalHit(true); // Set this internally to not run into problems
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
    }
    
    void OnCollisionEnter(Collision col)
    {
        // Enforce touching boxes in the beginning
        /*if (col.gameObject.CompareTag("wall"))
        {
            AddReward(-0.1f);
        }*/
    }
}


