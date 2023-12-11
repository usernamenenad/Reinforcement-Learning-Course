# A maze

It is possible to create everything essential for a maze using only your terminal! 

Firstly, I'd like you to execute in your terminal

```powershell
python main.py --help
```

just to check all options and how to use them. The usage example is shown below.

---

Before anything, create a base by executing in your terminal

```powershell
python main.py --base *BASE TYPE* --size *BASE SIZE*
```

It will ask you to type a name for a base. After that, create an environment 

```powershell
python main.py --environment *GAMMA VALUE*
```

Choose the base for your environment (which should be already created) and give your environment a name or just press enter, in which case the default name will be assigned.

Now, it is time to create an agent 

```powershell
python main.py --agent 
```

Like before, you should choose already existing environment and give your agent a name.

You're finally ready to compute V values. Do that by executing 

```powershell
python main.py --compute 
```

where you'll be asked for what agent should V values be executed.

If you want to display any info, do that by 

```powershell
python main.py --info *INFO FOR*
```
