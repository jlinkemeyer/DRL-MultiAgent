using System;
using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.UIElements;
using Random = UnityEngine.Random;

public class MacAgentBasicObservation : Agent
{
    public GameObject ground;
    public GameObject area;
    public GameObject goalA;
    public GameObject goalB;
    public GameObject goalC;
    public GameObject goalD;
    public GameObject wallA;
    public GameObject wallB;
    public GameObject wallC;
    public GameObject wallD;
    public GameObject instructionA;
    public GameObject instructionB;
    public GameObject instructionC;
    public GameObject instructionD;
    public Material materialRed;
    public Material materialBlue;
    public GameObject[] blocks;
    public bool useVectorObs;
    public GameObject middleObject;
    Rigidbody m_AgentRb;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    MacSettings m_MACSettings;
    int m_Selection;
    StatsRecorder m_statsRecorder;
    
    [HideInInspector]
    public MacGoalDetectBasicObservation macGoalDetect;

    private Vector3[] blockPositionMemory;
    private Vector3 agentPositionMemory;
    private GameObject[] goals;
    private GameObject[] instructions;
    private GameObject[] walls;
    private int normalize;

    // TODO
    public override void Initialize()
    {
        // Memorize the original block and agent positions for episode reset
        blockPositionMemory = new Vector3[blocks.Length];
        for (int i = 0; i < blocks.Length; i++)
        {
            blockPositionMemory[i] = blocks[i].transform.position;
            macGoalDetect = blocks[i].GetComponent<MacGoalDetectBasicObservation>();
            macGoalDetect.agent = this;
        }
        agentPositionMemory = this.transform.position;
        
        // Get all goals and instructions
        goals = new GameObject[] {goalA, goalB, goalC, goalD};
        walls = new GameObject[] {wallA, wallB, wallC, wallD};
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
            // The agents position and rotation
            /*sensor.AddObservation((this.transform.position.x - middleObject.transform.position.x) / 3.25f); // 1
            sensor.AddObservation((this.transform.position.z - middleObject.transform.position.z) / 9.25f); // 1
            sensor.AddObservation(this.transform.rotation.y / 360f); // 1

            // Positions and color of the red and blue boxes relative to agent
            foreach (var block in blocks)
            {
                if (block.active)
                {
                    if (block.CompareTag("blockRed")) // 1 * 2
                    {
                        sensor.AddObservation(0);
                    }
                    else
                    {
                        sensor.AddObservation(1);
                    }
                    sensor.AddObservation((block.transform.position.x - this.transform.position.x) / 6.25f); // 1 * 2
                    sensor.AddObservation((block.transform.position.z - this.transform.position.z) / 18.25f); // 1 * 2
                    //sensor.AddObservation(block.GetComponent<Rigidbody>().velocity); // 3
                }
            }

            // Instruction color
            foreach (var instruction in instructions)
            {
                if (instruction.active)
                {
                    if (instruction.CompareTag("instructionRed")) // 1
                    {
                        sensor.AddObservation(0);
                    }
                    else
                    {
                        sensor.AddObservation(1);
                    }
                }
            }
            
            // Goal position relative to agent
            foreach (var goal in goals)
            {
                if (goal.active)
                {
                    sensor.AddObservation((goal.transform.position.x - this.transform.position.x) / 3.22f); // 1
                    sensor.AddObservation((goal.transform.position.z - this.transform.position.z) / 20.85f); // 1
                }
            }*/
            
            // Time tracking
            sensor.AddObservation(StepCount / (float)MaxStep); // 1
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
        /*Debug.Log("StartObservation");
        foreach (var obs in GetObservations())
        {
            Debug.Log(obs);
        }
        Debug.Log("EndObservation");*/
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
        AddReward(score / normalize);

        // Swap ground material for a bit to indicate we scored -> red for wrong combination, green for correct
        if (score < 0)
        {
            m_statsRecorder.Add("Goal/Wrong", 1, StatAggregationMethod.Sum);
            StartCoroutine(GoalScoredSwapGroundMaterial(m_MACSettings.failMaterial, 0.5f));
        }
        else
        {
            m_statsRecorder.Add("Goal/Correct", 1, StatAggregationMethod.Sum);
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
    
    public override void OnEpisodeBegin()
    {
        // Get a random goal:
        int rdGoal = 3; //Random.Range(0, 4);
        int numberGoals = 4;
        normalize = 1;
        for (int i = 0; i < numberGoals; i++)
        {
            // Determine color pattern (0 = no show, 1 = blue, 2 = red)
            int rd = Random.Range(1, 3);

            if (rd == 1)
            {
                goals[i].SetActive(true);
                walls[i].SetActive(false);
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
                walls[i].SetActive(false);
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
                walls[i].SetActive(true);
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
        
        // To observe in tensorboard
        m_statsRecorder.Add("Goal/Correct", 0, StatAggregationMethod.Sum);
        m_statsRecorder.Add("Goal/Wrong", 0, StatAggregationMethod.Sum);
    }
}


